sh shots/glosslist/zeroshot-glosslist/run.sh natu1246

for shots in 1 2 3 5 10 30 50
do    
    sh "shots/glosslist/fewshot-$shots-glosslist/run.sh" natu1246
done

sh shots/no-glosslist/zeroshot/run.sh natu1246

for shots in 1 2 3 5 10 30 50
do    
    sh "shots/no-glosslist/fewshot-$shots/run.sh" natu1246
done
