for lang in gitx1241 lezg1247 natu1246 uspa1245
do
    # sh shots/glosslist/zeroshot-glosslist/run.sh 

    for shots in 1 2 5 10 30 50
    do    
        sh "retrieval/word-recall/word-recall-$shots/run.sh" $lang
    done
done