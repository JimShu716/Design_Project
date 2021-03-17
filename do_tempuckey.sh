#=====TODO=========
trainCollection=msrvtt10ktrain
valCollection=msrvtt10kval
testCollection=msrvtt10ktest
concate=full

# Generate a vocabulary on the training set
./do_get_vocab.sh $trainCollection

# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py $trainCollection $valCollection $testCollection \
                                            --overwrite 1 \
                                            --max_violation \
                                            --text_norm \
                                            --visual_norm \
                                            --concate $concate \
                                            --batch_size 128 \
                                            --loss_fun mrl \
                                            --measure exp \
                                            
# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
#./do_test_dual_encoding_${testCollection}.sh
