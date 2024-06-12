# for lang in gitx1241 lezg1247 natu1246 uspa1245 
# do
#     # sh shots/glosslist/zeroshot-glosslist/run.sh 

#     for shots in 1 2 5 10 30 50
#     do    
#         sh "retrieval/word-precision/word-precision-$shots/run.sh" $lang
#     done
# done

retriever="morpheme-recall"


for lang in gitx1241 lezg1247 natu1246 uspa1245 
do
    # sh shots/glosslist/zeroshot-glosslist/run.sh 

    for shots in 1 2 5 10 30 50
    do   
        for seed in 0 1 2
        do 
            if [ -f "retrieval/${retriever}/${retriever}-${shots}/${lang}.unseg.command-r-plus.${seed}.metrics.json" ]; then
                echo "File already exists, skipping: retrieval/${retriever}/${retriever}-${shots}/${lang}.unseg.command-r-plus.${seed}.metrics.json"
            else
                sh "retrieval/${retriever}/${retriever}-$shots/run.sh" $lang $seed
            fi
        done
    done
done