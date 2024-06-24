import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader

from logger.logger import Logger

from .model.LaTr import LaTr
from .data.GenVQADataset import GenVQADataset
from .data.utils import adapt_ocr

from timeit import default_timer as timer
from tqdm import tqdm

import evaluation

from transformers import AutoTokenizer, AutoConfig
import itertools


class Executor():
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        print("---Initializing Executor---")
        self.mode = mode
        self.config = config
        self.evaltype = evaltype
        self.predicttype = predicttype
        if self.mode == "train":
            self._create_data_utils()       


            self.model_config = AutoConfig.from_pretrained(config.backbone_name)

            self.model_config.update({"max_2d_position_embeddings" : config.max_2d_position_embeddings,
                                "vit_model" : self.config.vit_model_name})

            self.model = LaTr(self.model_config)

            self.model = self.model.to(self.config.DEVICE)

            self.optim = torch.optim.Adam(self.model.parameters(), lr=config.LR, betas=config.BETAS, eps=1e-9)

            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.optim, total_iters = config.warmup_step)

            self.SAVE = config.SAVE
            self._create_dataloader()

            if os.path.isfile(os.path.join(self.config.SAVE_PATH, "last_ckp.pth")):
                print("###Load trained checkpoint ...")
                ckp = torch.load(os.path.join(self.config.SAVE_PATH, "last_ckp.pth"))
                print(f"\t- Last train epoch: {ckp['epoch']}")
                self.model.load_state_dict(ckp['state_dict'])
                self.optim.load_state_dict(ckp['optimizer'])
            
        if self.mode in ["eval", "predict"]:
            self.init_eval_predict_mode()

            self.model_config = AutoConfig.from_pretrained(config.backbone_name)

            self.model_config.update({"max_2d_position_embeddings" : config.max_2d_position_embeddings,
                                "vit_model" : self.config.vit_model_name})


            self.model = LaTr(self.model_config)
            self.model = self.model.to(self.config.DEVICE)

   

    def train(self):
        if not self.config.SAVE_PATH:
            folder = './models'
        else:
            folder = self.config.SAVE_PATH
        
        if not os.path.exists(folder):
            os.mkdir(folder)


        best_f1 = 0
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
                if best_f1 < f1:
                    best_f1 = f1
                    statedict = {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "epoch": epoch,
                    }

                    filename = f"best_ckp.pth"
                    torch.save(statedict, os.path.join(folder,filename))
                    print(f"!---------Saved {filename}----------!")

                lstatedict = {
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optim.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "epoch": epoch,
                        }

                lfilename = f"last_ckp.pth"
                torch.save(lstatedict, os.path.join(folder,lfilename))
        
        e_train_time = timer()
        print(f"\n# BEST RESULT:\n\tEpoch: {m_epoch}\n\tBest F1: {m_f1:.4f}")
        print(f"#----------- TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
        
    def evaluate(self):
        print("###Evaluate Mode###")

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, f"{self.evaltype}_ckp.pth")):
            print("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, f"{self.evaltype}_ckp.pth"))
            print(f"\t- Using {self.evaltype} train epoch: {ckp['epoch']}")
            self.model.load_state_dict(ckp['state_dict'])

        elif os.path.isfile(os.path.join('./models', f"{self.evaltype}_ckp.pth")):
            print("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join('./models', f"{self.evaltype}_ckp.pth"))
            print(f"\t- Using {self.evaltype} train epoch: {ckp['epoch']}")
            self.model.load_state_dict(ckp['state_dict'])
        
        else:
            print(f"(!) {self.evaltype}_ckp.pth is required (!)")
            return 
        
        with torch.no_grad():
            print(f'Evaluate val data ...')

            res = self._evaluate_metrics()
            print(res)
    
    def predict(self): 
        print("###Predict Mode###")
        if os.path.isfile(os.path.join(self.config.SAVE_PATH, f"{self.predicttype}_ckp.pth")):
            print("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, f"{self.predicttype}_ckp.pth"))
            print(f"\t- Using {self.predicttype} train epoch: {ckp['epoch']}")
            self.model.load_state_dict(ckp['state_dict'])

        elif os.path.isfile(os.path.join('./models', f"{self.predicttype}_ckp.pth")):
            print("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join('./models', f"{self.predicttype}_ckp.pth"))
            print(f"\t- Using {self.predicttype} train epoch: {ckp['epoch']}")
            self.model.load_state_dict(ckp['state_dict'])
        else:
            print(f"(!) {self.predicttype}_ckp.pth is required  (!)")
            return

        print("## START PREDICTING ... ")
        print(f'\t#PREDICTION:\n')

        if self.config.get_predict_score:
            results, scores = self._evaluate_metrics()
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
    
    def run(self):
        log = Logger("./terminal.txt")
        log.start()

        if self.mode =='train':
            if self.config.STEP_MODE:
                self._train_step()
            self.train()
        elif self.mode == 'eval':
            self.evaluate()
        elif self.mode == 'predict':
            self.predict()
        else:
            exit(-1)

        log.stop()
            
    def _create_data_utils(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        ocr_df = adapt_ocr(self.config.ocr_path)

        print("# Creating Datasets")
        
        self.train_data = GenVQADataset(base_img_path = self.config.base_img_path,
                                        qa_df = train_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        max_ocr = self.config.max_ocr,
                                        transform=None,
                                        batch_encode = 128,
                                        max_seq_length = self.config.max_q_length,
                                        max_answer_length = self.config.max_a_length)

        self.val_data = GenVQADataset(base_img_path = self.config.base_img_path,
                                        qa_df = val_qa_df,
                                        ocr_df = ocr_df,
                                        tokenizer = self.tokenizer,
                                        max_ocr = self.config.max_ocr,
                                        transform=None,
                                        batch_encode = 128,
                                        max_seq_length = self.config.max_q_length,
                                        max_answer_length = self.config.max_a_length)
    

    def _create_dataloader(self):
        print("# Creating DataLoaders")
       
        self.trainiter = DataLoader(dataset = self.train_data, 
                                    batch_size=self.config.BATCH_SIZE, 
                                    shuffle=True)
        self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.BATCH_SIZE)

    def init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name)

        if self.mode == "eval":
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = adapt_ocr(self.config.ocr_path)

            self.val_data = GenVQADataset(base_img_path = self.config.base_img_path,
                                            qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            tokenizer = self.tokenizer,
                                            max_ocr = self.config.max_ocr,
                                            transform=None,
                                            batch_encode = 128,
                                            max_seq_length = self.config.max_q_length,
                                            max_answer_length = self.config.max_a_length)
            
            self.val_answer = list(val_qa_df["answer"])
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.BATCH_SIZE)

        elif self.mode == "predict":
            print("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question", "answer", "filename"]]
        
            ocr_df = adapt_ocr(self.config.ocr_path)

            self.predict_data = GenVQADataset(base_img_path = self.config.base_img_path,
                                                qa_df = predict_qa_df,
                                                ocr_df = ocr_df,
                                                tokenizer = self.tokenizer,
                                                max_ocr = self.config.max_ocr,
                                                transform=None,
                                                batch_encode = 128,
                                                max_seq_length = self.config.max_q_length,
                                                max_answer_length = self.config.max_a_length)
            
            if self.config.get_predict_score:
                self.predict_answer = list(val_qa_df["answer"])
            else:
                self.predict_answer = None

            self.predictiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.BATCH_SIZE)

    
    def _train_epoch(self, epoch):
        self.model.train()
        losses = 0
        with tqdm(desc='Epoch %d - Training ' % epoch , unit='it', total=len(list(self.trainiter))) as pbar:
            for it, batch in enumerate(self.trainiter):
                decoder_attention_mask = batch['decoder_attention_mask'].to(self.config.DEVICE)
                labels = batch['labels'].type(torch.long).to(self.config.DEVICE)


                trg_input = labels[:, :-1]
                decoder_attention_mask = decoder_attention_mask[:, :-1]

                logits = self.model(pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                    bbox = batch['bbox'].to(self.config.DEVICE),
                                    input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    labels = trg_input,
                                    attention_mask = batch['attention_mask'].to(self.config.DEVICE),
                                    decoder_attention_mask = decoder_attention_mask,
                                    bbox_attention_mask=batch['bbox_attention_mask'].to(self.config.DEVICE) ,
                                    tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE))


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

                    decoder_attention_mask = batch['decoder_attention_mask'].to(self.config.DEVICE)
                    labels = batch['labels'].type(torch.long).to(self.config.DEVICE)


                    trg_input = labels[:, :-1]
                    decoder_attention_mask = decoder_attention_mask[:, :-1]

                    logits = self.model( pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                    bbox = batch['bbox'].to(self.config.DEVICE),
                                    input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    labels = trg_input,
                                    attention_mask = batch['attention_mask'].to(self.config.DEVICE),
                                    decoder_attention_mask = decoder_attention_mask,
                                    bbox_attention_mask=batch['bbox_attention_mask'].to(self.config.DEVICE) ,
                                    tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE))


                    trg_out = labels[:, 1:]

                    loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                    losses += loss.data.item()

                    pbar.set_postfix(loss=losses / (it + 1))
                    pbar.update()


        return losses / len(list(self.valiter))

    def _train_step(self):
        assert self.config.END_STEP is not None
        assert self.config.END_STEP > 0

        self.model.train()
        losses = 0
        current_step = 0
        with tqdm(desc='Training on steps... ' , unit='it', total=self.config.END_STEP) as pbar:
            while True:
                for it, batch in enumerate(self.trainiter):
                    decoder_attention_mask = batch['decoder_attention_mask'].to(self.config.DEVICE)
                    labels = batch['labels'].type(torch.long).to(self.config.DEVICE)


                    trg_input = labels[:, :-1]
                    decoder_attention_mask = decoder_attention_mask[:, :-1]

                    logits = self.model(pixel_values = batch['pixel_values'].to(self.config.DEVICE),
                                        bbox = batch['bbox'].to(self.config.DEVICE),
                                        input_ids = batch['input_ids'].to(self.config.DEVICE),
                                        labels = trg_input,
                                        attention_mask = batch['attention_mask'].to(self.config.DEVICE),
                                        decoder_attention_mask = decoder_attention_mask,
                                        bbox_attention_mask=batch['bbox_attention_mask'].to(self.config.DEVICE) ,
                                        tokenized_ocr=batch['tokenized_ocr'].to(self.config.DEVICE))


                    self.optim.zero_grad()

                    trg_out = labels[:, 1:]

                    loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                    loss.backward()

                    self.optim.step()

                    self.scheduler.step()
                    
                    losses += loss.data.item()

                    pbar.set_postfix(loss=losses / (it + 1))
                    pbar.update()

                    current_step += 1

                    if current_step == self.config.eval_after_steps:
                        eval_loss = self._evaluate()

                    if current_step >= self.config.END_STEP:
                        return losses / self.config.END_STEP
    
    def infer_post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.tokenizer.eos_token_id)])
            except:
                res.append(out)

        return res

    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with tqdm(desc='Inferring... ', unit='it', total=len(list(dataloader))) as pbar:
            with torch.no_grad():
                for it, batch in enumerate(dataloader):
                    pixel_values = batch['pixel_values'].to(self.config.DEVICE)
                    bbox = batch['bbox'].to(self.config.DEVICE)
                    input_ids = batch['input_ids'].to(self.config.DEVICE)
                    attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                    bbox_attention_mask = batch['bbox_attention_mask'].to(self.config.DEVICE)
                    tokenized_ocr = batch['tokenized_ocr'].to(self.config.DEVICE)

                    pred = self.model.generate( pixel_values,
                                                bbox,
                                                input_ids,
                                                attention_mask,
                                                bbox_attention_mask,
                                                tokenized_ocr,
                                                max_length = max_length)

                    decoded_preds += self.tokenizer.batch_decode(self.infer_post_processing(pred.tolist()), skip_special_tokens=True)

                    pbar.update()

        return decoded_preds


    def _evaluate_metrics(self):
        if self.mode == "predict":
            pred = self.infer(self.predictiter, self.config.max_predict_length)
        else:
            pred = self.infer(self.valiter, self.config.max_eval_length)

        answers_gt = [i.strip().lower() for i in self.val_answer]

        answers_gen = [[i.strip().lower()] for i in pred]

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