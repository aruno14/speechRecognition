for i in *.mp3;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  if [ -f "${name}.wav" ]; then
    echo "$FILE exists."
  else
      ffmpeg -i "$i" "${name}.wav"
  fi
done
