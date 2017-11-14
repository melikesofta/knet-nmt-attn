alias julia="/Applications/Julia-0.5.app/Contents/Resources/julia/bin/julia"
SF="data/words/first20k data/words/last10k"
TF="data/words/first20k data/words/last10k"
BS=1
H=50
E=3
G="data/words/last10k"
GT=$G
PREF="13-11-words2010"
FILE="${PREF}H${H}E${E}BS${BS}"
OUT="out/$FILE"
GEN="gen/$FILE"
rm $GEN $OUT
time julia train.jl --sourcefiles $SF --targetfiles $TF --batchsize $BS --hidden $H --embedding $H --epochs $E --generate $G --generatedfile $GEN > $OUT;
perl tools/multi-bleu.perl $TF < $GEN >> $OUT
cat $OUT | sed '$!d' && say done