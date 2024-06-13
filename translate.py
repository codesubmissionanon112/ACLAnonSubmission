import evaluate
import pandas as pd
from datasets import load_dataset
#import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from transformers import BertGenerationDecoder, BertGenerationEncoder, EncoderDecoderModel, BertTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataCleaning import init_emo
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from statistics import mean
from sklearn.model_selection import train_test_split
from datasets import Dataset
twitEmo = load_dataset("dair-ai/emotion")
train = twitEmo["train"].to_pandas()
test = twitEmo["test"].to_pandas()
val = twitEmo["validation"].to_pandas()
twitEmo = pd.concat([train, test, val], ignore_index=True) 

emData = load_dataset("empathetic_dialogues")
train = emData["train"].to_pandas()
test = emData["test"].to_pandas()
val = emData["validation"].to_pandas()
emData = pd.concat([train, test, val], ignore_index=True) 

def trainEmoData(emotion):
    emotionTw  = {"afraid" : 4, "angry" : 3, "joyful" : 1, "sad" : 0, "surprised" : 5, "love" : 2}
   
    emotionData = emData[ (emData["context"]== emotion)  ]
    notEmotionData = emData[ (emData["context"]!= emotion)   ] 
   
    emotionDataT = twitEmo[ (twitEmo["label"] == emotionTw[emotion])  ]
    notEmotionDataT = twitEmo[ (twitEmo["label"] !=emotionTw[emotion])  ]
    target = [*emotionData["utterance"].to_list(), *emotionDataT["text"].to_list()]
    notTarget = [*notEmotionData["utterance"].to_list(), *notEmotionDataT["text"].to_list()]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(notTarget , show_progress_bar=True, convert_to_tensor=True)
    

    bothEmo = []
    print("utterance!", len(target))
    for utterance in target:
        question_embedding = model.encode(utterance, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, corpus_embeddings)
        hits = hits[0]
        otherEmo = notTarget[hits[0]["corpus_id"]] 
        #emotion, similar to
        bothEmo.append([utterance, otherEmo ])

    
    
    data = pd.DataFrame(bothEmo, columns= ["target", "other"])
    train, test = train_test_split(data, test_size= .2 )
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, test
   
    

 
def preprocess_function(examples):
    #https://huggingface.co/docs/transformers/tasks/translation
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    
    inputs =list(examples["other"] ) 
    targets =list(examples["target"]) #[ for example in examples]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def tune(train, test, emotion):
   

    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

    
    test = Dataset.from_pandas(test)
    train = Dataset.from_pandas(train)

    tokenized_test = test.map(preprocess_function, batched=True)

    tokenized_train = train.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="microsoft/GODEL-v1_1-base-seq2seq")
    

    training_args = Seq2SeqTrainingArguments(
    output_dir="models/test_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)
    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
    trainer.train()
    trainer.save_model("models/" + emotion + "_model")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels    


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    metric = evaluate.load("bleu")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    result = {"bleu": result['bleu']}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def loadTranslator(translatorver):
    modelFiles = {"anger" : "models/angry_model", "fear" : "models/afraid_model", "joy" : "models/joyful_model", "surprise" : "models/surprised_model", "sadness" : "models/sad_model" }
    model = AutoModelForSeq2SeqLM.from_pretrained(modelFiles[translatorver])
    return model
 
def handleAgent(userIn, agentEmo, vectoriser, model):
   
    inEmo = model.predict(vectoriser.transform([userIn])) #double check format of this value
    inEmo = inEmo.tolist()[0]
   
    #modify agent value 
    agentEmo = [mean([agentEmo[0], inEmo[0]]), mean([agentEmo[1], inEmo[1]]), mean([agentEmo[2], inEmo[2]]) ]
        
    #figure out which tuned to send to, nearest neighbor
    emovVals = [[0.167, 0.865, 0.657], [0.073, 0.840, 0.293], [0.980, 0.824, 0.794], [0.875, 0.875, 0.562], [0.225, 0.333, 0.149]] # [0.052, 0.775, 0.317],
    emovLabels = ["anger" , "fear" , "joy"  , "surprise" , "sadness" ]#, "disgust"
    
    neighbors = NearestNeighbors()
    neighbors.fit(emovVals)
    nearest = neighbors.kneighbors([agentEmo],1, return_distance= False)
    translatorver = emovLabels[nearest[0][0]]
    
    return agentEmo, translatorver


def genRes(userIn, model, tokenizer):
    inputs = tokenizer(userIn, return_tensors="pt", return_attention_mask=False, max_length=128, truncation=True)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

    return text

def trainEmoVec():
    test, train = init_emo()
    vectoriser = TfidfVectorizer()
    train["text"] =  train["text"].apply( lambda x: str(x) )
    test["text"] =  test["text"].apply( lambda x: str(x) )
    trainvectors = vectoriser.fit_transform(train["text"])
    trainTarget = train.drop(["text"], axis=1)
   
    validvectors = vectoriser.transform(test["text"])
    validTarget = test.drop(["text"], axis= 1)

    #save vectoriser 
    with open("models/vectoriser.pkl", "wb") as file:
        pickle.dump(vectoriser, file)
    #train multi
    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(trainvectors, trainTarget)
    #eval multi
    pred = model.predict(validvectors)
    print("absolute", mean_absolute_error(validTarget, pred ))
    print("squared", mean_absolute_error(validTarget, pred ))
    print("absolute percentage", mean_absolute_error(validTarget, pred ))

    #save model
    with open("models/vadAssigner.pkl", "wb") as file:
            pickle.dump(model, file)

def loadpickle(fileName):
    with open(fileName, "rb") as file:
        model = pickle.load(file)
    return model


def runModelInit():
    trainEmoVec()#sets vals and vectoriser models


    emoList = ["afraid", "angry", "joyful", "sad",  "surprised" ]#"disgusted",
    for emotion in emoList:
        train, test = trainEmoData(emotion)
        print(test)
        print(train)  
        tune(train, test, emotion)#will need to run for each emo ver

class Runtime():
    def __init__(self):
        self.angerModel = loadTranslator("anger")
        self.fearModel = loadTranslator("fear")
        self.joyModel = loadTranslator("joy")
        self.surpriseModel = loadTranslator("surprise")
        self.sadnessModel = loadTranslator("sadness")

        self.initialResmodel = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        self.initialRestokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

        self.translateTokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        self.agentEmo = [.875, .1, .282] #value for calm

        self.VADvectoriser = loadpickle("models/vectoriser.pkl")
        self.VADmodel = loadpickle("models/vadAssigner.pkl")
    
        self.evalutterance,self.evalures = setEvalData()

    def run(self, utterance):
    # =
        modelSelect = {"anger" : self.angerModel, "fear" : self.fearModel, "joy" : self.joyModel, "surprise" : self.surpriseModel, "sadness" : self.sadnessModel }

        initRes = genRes(utterance, self.initialResmodel, self.initialRestokenizer )
        self.agentEmo, translatorVer = handleAgent(utterance, self.agentEmo, self.VADvectoriser, self.VADmodel )
        
        model = modelSelect[translatorVer] 

        input_ids = self.translateTokenizer(
            initRes, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        outputs = model.generate(input_ids)
        emoRes = self.translateTokenizer.decode(outputs[0])
        return emoRes
    
 
    def evalBB(self):
        modelRes = []
        modelResAll = []
        for convo in self.evalutterance:
            convoRes = []
            for utterance in convo:
                reply = genRes(utterance, self.initialResmodel, self.initialRestokenizer )
                convoRes.append(reply)
                modelResAll.append(reply)
                
        modelRes.append(convoRes)
        print(len(self.evalutterance))
        print(len(modelRes))
        evalData = pd.DataFrame(modelResAll, columns= ["reply"])    
        evalData.to_csv("data/evalbb.tsv", sep="\t", index=False) 
        return modelRes

    def evalUntuned(self):
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        modelResAll = []
        utteranceAll = []
        modelRes = []
        for convo in self.evalutterance:
            #print(convo)
            convoRes = []
            for utterance in convo:
                reply = genRes(utterance, self.initialResmodel, self.initialRestokenizer )
                input_ids =tokenizer(
                    reply, add_special_tokens=False, return_tensors="pt"
                ).input_ids

                outputs = model.generate(input_ids)
                unTuned = self.translateTokenizer.decode(outputs[0])
                convoRes.append(unTuned)
                modelResAll.append(unTuned)
                utteranceAll.append(utterance)
                
        modelRes.append(convoRes)
        print(len(self.evalutterance))
        print(len(modelRes))
        evalData = pd.DataFrame(modelResAll, columns= ["reply"])    
        evalData.to_csv("data/evaluntuned.tsv", sep="\t", index=False) 
        evalu = pd.DataFrame(utteranceAll, columns= ["utterance"])    
        evalu.to_csv("data/evalutterances.tsv", sep="\t", index=False) 
        return modelRes



    def eval(self):
        
        modelResAll = []
        modelRes = []
        for convo in self.evalutterance:
            #print(convo)
            convoRes = []
            for utterance in convo:
                reply = self.run(utterance)
                convoRes.append(reply)
                modelResAll.append(reply)

                
            self.agentEmo = [.875, .1, .282] #reset for new convo


        modelRes.append(convoRes)
        print(len(self.evalutterance))
        print(len(modelRes))
        evalData = pd.DataFrame(modelResAll, columns= ["reply"])    
        evalData.to_csv("data/evalTranslate.tsv", sep="\t", index=False) 
        return modelRes
        
   
    def runText(self):   

       
        convo = ["I really want to take a nap. I feel very sleepy today.","I fell asleep very late. It was almost two o'clock in the morning when I finally fell asleep. "]
        humanRes = [" What's the matter? Didn't you get enough sleep last night?","Are you worried about something? Why couldn't you sleep?"]

        tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
        translateResEmos = []
        agentEmos = []
        agentEmos.append("Agent Emotion")
        translateRes = []
        translateRes.append("translate")
        translateResEmos.append("translate")
        for utterance in convo:
            reply = self.run(utterance)
            translateRes.append(reply)
            agentEmos.append(self.agentEmo)
            inEmo = self.VADmodel.predict(self.VADvectoriser.transform([reply])) #double check format of this value
            inEmo = inEmo.tolist()[0]
            translateResEmos.append(inEmo)

        bbResEmos = []
        bbRes = []
        bbRes.append("bb")
        bbResEmos.append("bb")
        for utterance in convo:
            reply = genRes(utterance, self.initialResmodel, self.initialRestokenizer )

            bbRes.append(reply)
            inEmo = self.VADmodel.predict(self.VADvectoriser.transform([reply])) #double check format of this value
            inEmo = inEmo.tolist()[0]
            bbResEmos.append(inEmo)

        unTunedResEmos = []
        unTunedRes = []
        unTunedRes.append("unTuned")
        unTunedResEmos.append("unTuned")
        for utterance in convo:
            reply = genRes(utterance, self.initialResmodel, self.initialRestokenizer )
            input_ids =tokenizer(
                    reply, add_special_tokens=False, return_tensors="pt"
                ).input_ids

            outputs = model.generate(input_ids)
            unTuned = self.translateTokenizer.decode(outputs[0])
            unTunedRes.append(unTuned)
            inEmo = self.VADmodel.predict(self.VADvectoriser.transform([reply])) #double check format of this value
            inEmo = inEmo.tolist()[0]
            unTunedResEmos.append(inEmo)

        evalData = pd.DataFrame([translateRes, bbRes, unTunedRes], columns= ["model", "res1", "res2"])    
        evalData.to_csv("data/evalResText.tsv", sep="\t", index=False) 
        evalDataemo = pd.DataFrame([agentEmos, translateResEmos, bbResEmos, unTunedResEmos], columns= ["model", "res1Emo", "res2Emo"])    
        evalDataemo.to_csv("data/evalResTextEmos.tsv", sep="\t", index=False) 

        #111 traindataset
        #Person1#: I really want to take a nap. I feel very sleepy today. #Person2#: What's the matter? Didn't you get enough sleep last night? #Person1#: I fell asleep very late. It was almost two o'clock in the morning when I finally fell asleep. #Person2#: Are you worried about something? Why couldn't you sleep? #Person1#: You know how it is when you're in a strange country. Everything is new, and you get tired and nervous sometimes. Then you worry about your family, about conditions back home, about your courses, about your money, about everything. I tried to fall asleep but I just had too much on my mind. #Person2#: Well, take it easy. Things will look better tomorrow. Maybe you should try exercising or a hot bath to help you relax. #Person1#: Anything is worth a try. But right now I really just want to find a quiet place to take a nap.
       
#python translate.py
def setEvalData():
    dataset = load_dataset("knkarthick/dialogsum")
    dataset = dataset["validation"].to_pandas()

    dialogs =dataset["dialogue"].tolist()#list(set()) 
    utterancesList = []
    utterancesListcheck = []
    resList = []
    for log in dialogs:
        if "3#:" not in log:
            utterances = []
            reply = []
            
            l = log.split("#Person")
            for utterance in l:
                #print(utterance)
                utterance = utterance.replace("\n"," ")
                utterance = utterance.replace(chr(34),"")
                if "1#" in utterance :
                    utterance = utterance.replace("1#:"," ")
                    utterances.append(utterance)
                    utterancesListcheck.append(utterance)
                elif "2#" in utterance :
                    utterance = utterance.replace("2#:"," ")
                    resList.append(utterance)
                    reply.append(utterance)
            if len(utterances)  != len(reply):
                print(utterances, reply)
                print(len(resList))#del utterances[-1]
                resList = resList[:-len(reply)]
                utterancesListcheck = utterancesListcheck[:-len(utterances)]
                print(len(resList))
                
            else: 
                utterancesList.append(utterances)   
            
            print(len(flatten(utterancesList)), len(resList))
    evalData = pd.DataFrame(resList, columns= ["reply"])
    evalData.to_csv("data/replyEvalSet.tsv",sep="\t", index=False)    
    return utterancesList, resList
def evalMetrics():
    evalIn = pd.read_csv("data/replyEvalSet.tsv", sep="\t")
    translate = pd.read_csv("data/evalTranslate.tsv", sep="\t")
    untuned = pd.read_csv("data/evaluntuned.tsv", sep="\t")
    bb = pd.read_csv("data/evalbb.tsv", sep="\t")

    for model in [translate, untuned, bb ]:
        for y in ["<pad>", "</s>", "<s>" ]:
            model["reply"] = model["reply"].apply(lambda x : x.replace(y,""))
        print(model["reply"])
        
    print(evalIn["reply"])
    evalIn = evalIn["reply"].tolist()
   
    
    modelRes = {"translate" : translate["reply"].tolist() , "untuned" : untuned["reply"].tolist() , "bb" : bb["reply"].tolist() }
   
    metricResAll = []
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    bleu = evaluate.load("bleu")
  
    for ver in modelRes:
        metricRes = []
        metricRes.append(ver)
        predictions = modelRes[ver] # don't forget to remove padding tokens

        rougeresults = rouge.compute(predictions= predictions, references= evalIn  )
        metricRes.append(rougeresults['rougeL'])
        
        print(rougeresults)
        meteorresults = meteor.compute(predictions= predictions, references= evalIn)
        metricRes.append(meteorresults['meteor'])

        bleuresults = bleu.compute(predictions= predictions, references= evalIn)
        metricRes.append(bleuresults['bleu'])

        metricResAll.append(metricRes)
    print(metricResAll)
    evalData = pd.DataFrame(metricResAll, columns= ["Approach",'rougeL', 'meteor', 'bleu' ])
    evalData.to_csv("data/metrics.tsv",sep="\t", index=False)    

def flatten(xss):
    return [x for xs in xss for x in xs]


runModelInit()
r = Runtime()
r.runText()
r.evalBB()
r.eval()
r.evalUntuned()
evalMetrics()
