#!/usr/bin/env bash
# source:
# https://github.com/matthuisman/files.matthuisman.nz/blob/master/gdrivedl

url="https://drive.google.com/file/d/1nmjDoBwxuTViOvAVJAsGOqm_a8rO4TQe/view?usp=sharing"
filename="questions.zip"

[ -z "$url" ] && echo A URL or ID is required first argument && exit 1

fileid=""
declare -a patterns=("s/.*\/file\/d\/\(.*\)\/.*/\1/p" "s/.*id\=\(.*\)/\1/p" "s/\(.*\)/\1/p")
for i in "${patterns[@]}"
do
   fileid=$(echo $url | sed -n $i)
   [ ! -z "$fileid" ] && break
done

[ -z "$fileid" ] && echo Could not find Google ID && exit 1

echo File ID: $fileid 

tmp_file="$filename.$$.file"
tmp_cookies="$filename.$$.cookies"
tmp_headers="$filename.$$.headers"

url='https://docs.google.com/uc?export=download&id='$fileid
echo Downloading: "$url > $tmp_file"
wget --save-cookies "$tmp_cookies" -q -S -O - $url 2> "$tmp_headers" 1> "$tmp_file"

if [[ ! $(find "$tmp_file" -type f -size +10000c 2>/dev/null) ]]; then
   confirm=$(cat "$tmp_file" | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
fi

if [ ! -z "$confirm" ]; then
   url='https://docs.google.com/uc?export=download&id='$fileid'&confirm='$confirm
   echo Downloading: "$url > $tmp_file"
   wget --load-cookies "$tmp_cookies" -q -S -O - $url 2> "$tmp_headers" 1> "$tmp_file"
fi

[ -z "$filename" ] && filename=$(cat "$tmp_headers" | sed -rn 's/.*filename=\"(.*)\".*/\1/p')
[ -z "$filename" ] && filename="google_drive.file"

echo Moving: "$tmp_file > $filename"

mv "$tmp_file" "$filename"

rm -f "$tmp_cookies" "$tmp_headers"

echo Saved: "$filename"

unzip questions.zip
rm questions.zip
echo "Done"

exit 0