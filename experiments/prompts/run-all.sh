for retriever in chrf
do
    for lang in gitx1241 lezg1247 natu1246 uspa1245 
    do 
        if [ -f "./prompts/${retriever}/${lang}.unseg.command-r-plus.${seed}.metrics.json" ]; then
            echo "File already exists, skipping: prompts/${retriever}/${lang}.unseg.command-r-plus.0.metrics.json"
        else
            sh "./prompts/${retriever}/run.sh" $lang
        fi
    done
done