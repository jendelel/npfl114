prefix="dev"
id_file="${prefix}_ids.txt"
ref_label_file="essay_${prefix}_labels.txt"
out_label_file="${prefix}_predictions.txt"

paste $id_file $out_label_file > "pasted.txt"
python pasted2predictions.py
out_labels="labels.txt"
lines="`cat $out_labels | wc -l`"
same="`paste $out_labels $ref_label_file | awk '{if ($1 == $2) {print $0}}' | wc -l`"
echo "Lines: $lines"
echo "Same: $same"
echo "Accuracy: `echo "scale=2; $same/$lines*100" | bc`"

