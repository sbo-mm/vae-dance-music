#!/usr/bin/env bash

CAMERA="$1"
AIST_FOLDER="../aist_db"
FILE_POSTFIX="/*.mp4" 
#LIST="refined_2M_all_video_url.csv"; *${CAMERA}

OUT_FILE=""
OUT_FILE_PREFIX="4sec_"
TMP_FILE1="out_temp_1.mp4"
TMP_FILE2="out_temp_2.mp4"

for path in $(find "$AIST_FOLDER" -name "*${CAMERA}*") ; do
	echo "Processing ${path}"
	ffmpeg -y -i "$path" -movflags faststart "$TMP_FILE1" &>/dev/null
	ffmpeg -y -ss "00:00:00" -i "$TMP_FILE1" -t "00:00:04" -map 0 "$TMP_FILE2" &>/dev/null
	FILE=$(basename "$path")
	OUT_FILE="${OUT_FILE_PREFIX}${FILE}"
	ffmpeg -y -i "$TMP_FILE2" -vf "scale=480:-1" "$OUT_FILE" &>/dev/null
	echo "Saving to ${OUT_FILE}"
done 

rm "TMP_FILE1"
rm "TMP_FILE1"
