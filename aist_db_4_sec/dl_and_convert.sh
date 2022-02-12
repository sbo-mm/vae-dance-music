#!/bin/sh

logfile=encodemp4ize.log
echo "Started at $(date)." > "$logfile"
rsync -avz --exclude '*.mp4' ../aist_db ./

find ../aist_db -type f -name '*.mp4' -exec sh -c '
for mp4file; do
	file=${mp4file#../aist_db/} 
	< /dev/null ffmpeg -y -i "$mp4file" -movflags faststart temp_file_1.mp4
	< /dev/null ffmpeg -y -ss "00:00:00" -i temp_file_1.mp4 -t "00:00:04" -map 0 temp_file_2.mp4
	< /dev/null ffmpeg -y -i temp_file_2.mp4 -vf "scale=480:-1" "./${file%mp4}"mp4
	printf %s\\n "$mp4file MP4 done." >> "$logfile"
done
' _ {} +

