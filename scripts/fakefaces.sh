if [ $# -eq 0 ]
   then
      echo "./fakefaces.sh [number of samples]"
      exit 1
fi

mkdir -p ../data/FakeFaces

for ((i=0; i < $1; i++))
do
    now=`date +"%s"`
    curl 'https://thispersondoesnotexist.com/image' \
  -H 'authority: thispersondoesnotexist.com' \
  -H 'dnt: 1' \
  -H 'if-none-match: "6019a64e-7e589"' \
  -H 'if-modified-since: Tue, 02 Feb 2021 19:21:50 GMT' \
  -H 'user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36' \
  -H 'accept: image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8' \
  -H 'sec-fetch-site: same-origin' \
  -H 'sec-fetch-mode: no-cors' \
  -H 'sec-fetch-dest: image' \
  -H 'referer: https://thispersondoesnotexist.com/' \
  -H 'accept-language: en-US,en;q=0.9' \
  -H 'cookie: __cfduid=d4842e038ddca841075a83272e9bd6b331612293555' \
  --compressed --output "../data/FakeFaces/${now}.jpg"
done
