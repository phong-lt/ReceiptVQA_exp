import os
import sys
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader

from logger.logger import Logger

from core.model import PlainT5
from core.data import TextOnlyVQADataset, textonly_ocr_adapt

from timeit import default_timer as timer
from tqdm import tqdm

import evaluation

from transformers import AutoTokenizer, AutoConfig
import itertools


class PlainT5Executor():
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        print("---Initializing Executor---")
        self.mode = mode
        self.config = config
        self.evaltype = evaltype
        self.predicttype = predicttype
        self.best_score = 0

        if self.mode == "train":
            self._create_data_utils()        
            self._build_model()
            self._create_dataloader()

            self.optim = torch.optim.Adam(self.model.parameters(), lr=config.LR, betas=config.BETAS, eps=1e-9)
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)    
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.optim, total_iters = config.warmup_step)

            self.SAVE = config.SAVE

            if os.path.isfile(os.path.join(self.config.SAVE_PATH, "last_ckp.pth")):
                print("###Load trained checkpoint ...")
                ckp = torch.load(os.path.join(self.config.SAVE_PATH, "last_ckp.pth"))
                try:
                    print(f"\t- Last train epoch: {ckp['epoch']}")
                except:
                    print(f"\t- Last train step: {ckp['step']}")
                self.model.load_state_dict(ckp['state_dict'])
                self.optim.load_state_dict(ckp['optimizer'])
                self.scheduler.load_state_dict(ckp['scheduler'])
                self.best_score = ckp['best_score']
            
        if self.mode in ["eval", "predict"]:
            self._init_eval_predict_mode()
            self._build_model()

    def run(self):
        log = Logger("./terminal.txt")
        log.start()

        if self.mode =='train':
            if self.config.STEP_MODE:
                print("# Training on steps... #")
                self._train_step()
            else:
                print("# Training on epochs... #")
                self.train()
        elif self.mode == 'eval':
            self.evaluate()
        elif self.mode == 'predict':
            self.predict()
        else:
            exit(-1)

        log.stop()

    def train(self):
        if not self.config.SAVE_PATH:
            folder = './models'
        else:
            folder = self.config.SAVE_PATH
        
        if not os.path.exists(folder):
            os.mkdir(folder)

        m_f1 = 0
        m_epoch = 0

        print(f"#----------- START TRAINING -----------------#")
        s_train_time = timer()

        for epoch in range(1, self.config.NUM_EPOCHS+1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._evaluate()
            res = self._evaluate_metrics()
            f1 = res["F1"]
            print(f'\tTraining Epoch {epoch}:')
            print(f'\tTrain Loss: {train_loss:.4f} - Val. Loss: {val_loss:.4f}')
            print(res)
            
            if m_f1 < f1:
                m_f1 = f1
                m_epoch = epoch

            if self.SAVE:
                if self.best_score < f1:
                    self.best_score = f1
                    statedict = {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "best_score": self.best_score
                    }

                    filename = f"best_ckp.pth"
                    torch.save(statedict, os.path.join(folder,filename))
                    print(f"!---------Saved {filename}----------!")

                lstatedict = {
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optim.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "epoch": epoch,
                            "best_score": self.best_score
                        }

                lfilename = f"last_ckp.pth"
                torch.save(lstatedict, os.path.join(folder,lfilename))
        
        e_train_time = timer()
        if m_f1 < self.best_score:
            m_f1 = self.best_score
            m_epoch = -1
        print(f"\n# BEST RESULT:\n\tEpoch: {m_epoch}\n\tBest F1: {m_f1:.4f}")
        print(f"#----------- TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
        
    def evaluate(self):
        print("###Evaluate Mode###")

        self._load_trained_checkpoint(self.evaltype)
        
        with torch.no_grad():
            res = self._evaluate_metrics()
            print(f'\t#EVALUATION:\n')
            print(res)
    
    def predict(self): 
        print("###Predict Mode###")
        
        self._load_trained_checkpoint(self.predicttype)

        print("## START PREDICTING ... ")

        if self.config.get_predict_score:
            results, scores = self._evaluate_metrics()
            print(f'\t#PREDICTION:\n')
            print(f'\t{scores}')
        else:
            preds = self.infer(self.predictiter, self.config.max_predict_length)
            results = [{"gens": p} for p in preds]



        if self.config.SAVE_PATH:
            with open(os.path.join(self.config.SAVE_PATH, "results.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print("Saved Results !")
        else:
            with open(os.path.join(".","results.csv"), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print("Saved Results !")

    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with tqdm(desc='Inferring... ', unit='it', total=len(list(dataloader))) as pbar:
            with torch.no_grad():
                for it, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(self.config.DEVICE)
                    pred = self.model.generate( input_ids,
                                                max_length = max_length)

                    decoded_preds += self.tokenizer.batch_decode(self._infer_post_processing(pred.tolist()), skip_special_tokens=True)

                    pbar.update()

        return decoded_preds
    
    def _build_model(self):
        if self.config.MODEL_MOD_CONFIG_CLASS is not None:   
            self.model_config = self.build_class(self.config.MODEL_MOD_CONFIG_CLASS)().build(self.config)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.backbone_name)

        self.model = self.build_class(self.config.MODEL_CLASS)(self.model_config)
        self.model = self.model.to(self.config.DEVICE)
    
    def _load_trained_checkpoint(self, loadtype):

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, f"{loadtype}_ckp.pth")):
            print("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, f"{loadtype}_ckp.pth"))
            try:
                print(f"\t- Using {loadtype} train epoch: {ckp['epoch']}")
            except:
                print(f"\t- Using {loadtype} train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])

        elif os.path.isfile(os.path.join('./models', f"{loadtype}_ckp.pth")):
            print("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join('./models', f"{loadtype}_ckp.pth"))
            try:
                print(f"\t- Using {loadtype} train epoch: {ckp['epoch']}")
            except:
                print(f"\t- Using {loadtype} train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
        
        else:
            raise Exception(f"(!) {loadtype}_ckp.pth is required (!)")


    def _create_data_utils(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        ocr_df = textonly_ocr_adapt(self.config.ocr_path)

        print("# Creating Datasets")
        
        self.train_data = TextOnlyVQADataset(
                                        qa_df = train_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        batch_process = 128,
                                        max_input_length = self.config.max_input_length,
                                        max_output_length = self.config.max_output_length)

        self.val_data = TextOnlyVQADataset(
                                        qa_df = val_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        batch_process = 128,
                                        max_input_length = self.config.max_input_length,
                                        max_output_length = self.config.max_output_length)
    

    def _create_dataloader(self):
        print("# Creating DataLoaders")
       
        self.trainiter = DataLoader(dataset = self.train_data, 
                                    batch_size=self.config.TRAIN_BATCH_SIZE, 
                                    shuffle=True)
        self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)

    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        if self.mode == "eval":
            print("###Load eval data ...")
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question_id", "question", "answer", "filename"]]
        
            ocr_df = textonly_ocr_adapt(self.config.ocr_path)

            self.val_data = TextOnlyVQADataset(
                                            qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            tokenizer = self.tokenizer,
                                            batch_process = 128,
                                            max_input_length = self.config.max_input_length,
                                            max_output_length = self.config.max_output_length)
            
            self.val_answer = list(val_qa_df["answer"])
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)

        elif self.mode == "predict":
            print("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question_id", "question", "answer", "filename"]]
        
            ocr_df = textonly_ocr_adapt(self.config.ocr_path)

            self.predict_data = TextOnlyVQADataset(
                                                qa_df = predict_qa_df,
                                                ocr_df = ocr_df,
                                                tokenizer = self.tokenizer,
                                                batch_process = 128,
                                                max_input_length = self.config.max_input_length,
                                                max_output_length = self.config.max_output_length)
            
            if self.config.get_predict_score:
                self.predict_answer = list(predict_qa_df["answer"])
            else:
                self.predict_answer = None

            self.predictiter = DataLoader(dataset = self.predict_data, 
                                    batch_size=self.config.PREDICT_BATCH_SIZE)

    
    def _train_epoch(self, epoch):
        self.model.train()
        losses = 0
        with tqdm(desc='Epoch %d - Training ' % epoch , unit='it', total=len(list(self.trainiter))) as pbar:
            for it, batch in enumerate(self.trainiter):
                decoder_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


                trg_input = labels[:, :-1]
                decoder_attention_mask = decoder_attention_mask[:, :-1]

                logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_ids = trg_input,
                                    label_attention_mask = decoder_attention_mask,)


                self.optim.zero_grad()

                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                loss.backward()

                self.optim.step()

                self.scheduler.step()
                
                losses += loss.data.item()

                pbar.set_postfix(loss=losses / (it + 1))
                pbar.update()

        return losses / len(list(self.trainiter))
    
    def _evaluate(self):
        self.model.eval()
        losses = 0
        with tqdm(desc='Validating... ' , unit='it', total=len(list(self.valiter))) as pbar:
            with torch.no_grad():
                for it, batch in enumerate(self.valiter):

                    decoder_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                    labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


                    trg_input = labels[:, :-1]
                    decoder_attention_mask = decoder_attention_mask[:, :-1]

                    logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                        src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                        label_ids = trg_input,
                                        label_attention_mask = decoder_attention_mask,)


                    trg_out = labels[:, 1:]

                    loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                    losses += loss.data.item()

                    pbar.set_postfix(loss=losses / (it + 1))
                    pbar.update()


        return losses / len(list(self.valiter))

    def _train_step(self):
        assert self.config.END_STEP is not None
        assert self.config.END_STEP > 0

        if not self.config.SAVE_PATH:
            folder = './models'
        else:
            folder = self.config.SAVE_PATH
        
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.train()

        losses = 0
        current_step = 0

        m_f1 = 0
        m_step = 0

        print(f"#----------- START TRAINING -----------------#")
        print(f"(!) Show train loss after each {self.config.show_loss_after_steps} steps")
        print(f"(!) Evaluate after each {self.config.eval_after_steps} steps")
        s_train_time = timer()

        while True:
            for batch in self.trainiter:
                decoder_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['label_ids'].type(torch.long).to(self.config.DEVICE)


                trg_input = labels[:, :-1]
                decoder_attention_mask = decoder_attention_mask[:, :-1]

                logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_ids = trg_input,
                                    label_attention_mask = decoder_attention_mask,)


                self.optim.zero_grad()

                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                loss.backward()

                self.optim.step()

                self.scheduler.step()
                
                losses += loss.data.item()

                current_step += 1

                if current_step % self.config.show_loss_after_steps == 0:
                    print(f"[Step {current_step} | {int(current_step/self.config.END_STEP*100)}% completed] Train Loss: {losses / current_step}")

                if current_step % self.config.eval_after_steps == 0:
                    eval_loss = self._evaluate()
                    res = self._evaluate_metrics()
                    f1 = res["F1"]
                    print(f'\tTraining Step {current_step}:')
                    print(f'\tTrain Loss: {losses / current_step} - Val. Loss: {eval_loss:.4f}')
                    print(res)
                    
                    if m_f1 < f1:
                        m_f1 = f1
                        m_step = current_step

                    if self.SAVE:
                        if self.best_score < f1:
                            self.best_score = f1
                            statedict = {
                                "state_dict": self.model.state_dict(),
                                "optimizer": self.optim.state_dict(),
                                "scheduler": self.scheduler.state_dict(),
                                "step": current_step,
                                "best_score": self.best_score
                            }

                            filename = f"best_ckp.pth"
                            torch.save(statedict, os.path.join(folder,filename))
                            print(f"!---------Saved {filename}----------!")

                        lstatedict = {
                                    "state_dict": self.model.state_dict(),
                                    "optimizer": self.optim.state_dict(),
                                    "scheduler": self.scheduler.state_dict(),
                                    "step": current_step,
                                    "best_score": self.best_score
                                }

                        lfilename = f"last_ckp.pth"
                        torch.save(lstatedict, os.path.join(folder,lfilename))

                if current_step >= self.config.END_STEP:
                    if m_f1 < self.best_score:
                        m_f1 = self.best_score
                        m_step = -1
                    e_train_time = timer()
                    print(f"\n# BEST RESULT:\n\tStep: {m_step}\n\tBest F1: {m_f1:.4f}")
                    print(f"#----------- TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
                    return
       
    def _infer_post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.tokenizer.eos_token_id)])
            except:
                res.append(out)

        return res

    def _evaluate_metrics(self):
        if self.mode == "predict":
            pred = self.infer(self.predictiter, self.config.max_predict_length)
            answers_gt = [i.strip() for i in self.predict_answer]
        else:
            pred = self.infer(self.valiter, self.config.max_eval_length)
            answers_gt = [i.strip() for i in self.val_answer]

        answers_gen = [[i.strip()] for i in pred]

        gens = {}
        gts = {}
        for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
            gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
            gens['%d_' % (i)] = [gen_i, ]
            gts['%d_' % (i)] = [gts_i]
    
        score, _ = evaluation.compute_scores(gts, gens)

        if self.mode == "predict":
            result = [{
                "gens": gen,
                "gts": gt 
            } for gen, gt in zip(answers_gen, answers_gt)]
            return result, score

        return score
    
    def build_class(self, classname):
        """
        convert string -> class
        """
        return getattr(sys.modules[__name__], classname)