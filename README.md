# BERT-BILSTM-GCN-CRF-for-NER
在原本BERT-BILSTM-CRF上融合GCN和词性标签等做NER任务
### 数据格式
        高 B-剧种 B-noun<br>
        腔 I-剧种 I-noun<br>
        ： O O<br>
        马 B-人名 B-noun<br>
平 I-人名 I-noun<br>
所 O O<br>
著 O B-verb<br>
扶 O B-verb<br>
贫 O I-verb<br> 
小 O B-noun<br>
说 O I-noun<br>
### 运行
python bert_gcn_ner.py
    --do_lower_case=False
    --do_train=False
    --do_eval=False
    --do_test=True
    --dataset=opera2-pos3
    --vocab_file=./chinese_L-12_H-768_A-12/vocab_update.txt
    --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json
    --init_checkpoint=./output3/bilstm_pos/model.ckpt-6720
    --max_seq_length=128
    --train_batch_size=32
    --learning_rate=2e-5 
    --num_train_epochs=10.0 
    --dropout_rate=0.5 
    --output_dir=./output3/bilstm_pos
    --bilstm=True
    --crf=True
    --use_pos=True 
    --gcn=1
