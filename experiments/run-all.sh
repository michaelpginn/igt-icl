# for lang in gitx1241 lezg1247 natu1246 uspa1245 
# do
#     # sh shots/glosslist/zeroshot-glosslist/run.sh 

#     for shots in 1 2 5 10 30 50
#     do    
#         sh "retrieval/word-precision/word-precision-$shots/run.sh" $lang
#     done
# done

for lang in gitx1241 lezg1247 natu1246 uspa1245 
do
    # sh shots/glosslist/zeroshot-glosslist/run.sh 

    for shots in 1 2 5 10 30 50
    do   
        for seed in 0 1 2
        do 
            if [ -f "retrieval/max-word-coverage/max-word-coverage-${shots}/${lang}.unseg.command-r-plus.${seed}.metrics.json" ]; then
                echo "File already exists, skipping: retrieval/max-word-coverage/max-word-coverage-${shots}/${lang}.unseg.command-r-plus.${seed}.metrics.json"
            else
                sh "retrieval/max-word-coverage/max-word-coverage-$shots/run.sh" $lang $seed
            fi
        done
    done
done