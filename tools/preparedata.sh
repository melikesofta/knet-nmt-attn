# data from http://app.aspell.net/create is used
# exclude words ending with 's
grep -v "^.*'s$" words70 > wordswithouts
# lowercase all words
../tools/mosesdecoder/scripts/tokenizer/lowercase.perl < wordswithouts > wordslowercasewithouts
# exlude words shorter than 6 chars
grep '^.\{6,25\}$' wordslowercasewithouts > longerwordswithouts
# sample 100k words out of the list
cat longerwordswithouts | gshuf -n 100000 > words100k
# put space between letters
sed -e 's/\(.\)/\1 /g' < words100k > words100kspaced