## Download ISRUC-S3 dataset
#  URL: https://sleeptight.isr.uc.pt/

mkdir -p ./data/ISRUC_S3/RawData
echo 'Make data dir: ./data/ISRUC_S3'

cd ./data/ISRUC_S3/RawData
for s in {1..10}
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
    unrar x $s.rar
    mv $s/$s.rec $s/$s.edf
done
echo 'Download Data to "./data/ISRUC_S3/RawData" complete.'
