mkdir ../Data

wget https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip && \
unzip TNBC_NucleiSegmentation.zip -d TNBC_NucleiSegmentation && \
mv TNBC_NucleiSegmentation/TNBC_NucleiSegmentation ../Data/ && \
rm -r TNBC_NucleiSegmentation && \
rm TNBC_NucleiSegmentation.zip

wget http://members.cbio.mines-paristech.fr/~pnaylor/Downloads/ForDataGenTrainTestVal.zip && \
unzip ForDataGenTrainTestVal.zip -d ForDataGenTrainTestVal && \
mv ForDataGenTrainTestVal/ForDataGenTrainTestVal ../Data/ && \
rm -r ForDataGenTrainTestVal && \
rm ForDataGenTrainTestVal.zip
rm -r ../Data/ForDataGenTrainTestVal/Slide_test
rm -r ../Data/ForDataGenTrainTestVal/GT_test

wget http://members.cbio.mines-paristech.fr/~pnaylor/Downloads/DataCPM.zip && \
unzip DataCPM.zip -d DataCPM && \
mv DataCPM/DataCPM ../Data/ && \
rm -r DataCPM && \
rm DataCPM.zip