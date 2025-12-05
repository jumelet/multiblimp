# Italian
echo -e "essere\tè\tV|IND;PRS;3;SG\tessere|X" >> ita/ita.segmentations

# French
grep -vP "^avoyr\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vP "^aveir\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vP "^savir\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vP "^saveir\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vP "^reçoivre\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vP "^estre\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vP "^estre\t" fra/fra.segmentations > fra/fra2; mv fra/fra2 fra/fra.segmentations
grep -vP "^mectre\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra

echo -e "être\test\tV|IND;PRS;3;SG\têtre|X" >> fra/fra.segmentations
echo -e "être\tsommes\tV|IND;PRS;1;PL\têtre|X" >> fra/fra.segmentations
echo -e "être\têtes\tV|IND;PRS;2;PL\têtre|X" >> fra/fra.segmentations
echo -e "être\tsont\tV|IND;PRS;3;PL\têtre|X" >> fra/fra.segmentations

echo -e "mettre\tmet\tV|IND;PRS;3;SG\têtre|X" >> fra/fra.segmentations

# English
sed -nE "s/([a-z]+)\t[a-z]+\tV;PRS;3;SG/\1\t\1\tV;PRS;3;PL/p" eng/eng >> eng/eng
sed "s/V;PRS;3/V;PRS;3;IND/" eng/eng > eng/eng2; mv eng/eng2 eng/eng
echo -e "be\tam\tV;IND;PRS;1;SG" >> eng/eng
echo -e "be\tare\tV;IND;PRS;2;SG" >> eng/eng
echo -e "be\tis\tV;IND;PRS;3;SG" >> eng/eng
echo -e "be\tare\tV;IND;PRS;1;PL" >> eng/eng
echo -e "be\tare\tV;IND;PRS;2;PL" >> eng/eng
echo -e "be\tare\tV;IND;PRS;3;PL" >> eng/eng

# Finnish
cp fin/fin.1 fin/fin

# Azerbaijani
grep -vP "(dövri|elektron|uçan|xarici)" aze/aze > aze/aze2; mv aze/aze2 aze/aze

# Uzbek
cat uzb/uzb_verbs >> uzb/uzb

# Livvi
cp olo/olo-new-written-livvic olo/olo
tail -n +2 olo/olo > olo/olo2; mv olo/olo2 olo/olo

# Karelian
cat krl/krl-new-written-karelian >> krl/krl
grep -v "#" krl/krl > krl/krl2; mv krl/krl2 krl/krl

# Veps
cp vep/vep-new-written-veps vep/vep

# Faroese
sed -E "s/\t(.)\.(.+)\.(.+)$/\t\1;\2;\3/" fao/fao > fao/fao2; mv fao/fao2 fao/fao
sed -E "s/\t(.)\.(.+)$/\t\1;\2/" fao/fao > fao/fao2; mv fao/fao2 fao/fao

# Korean
grep -vP "^[^\t]+\t[^\t]+\t$" kor/kor > kor/kor2; mv kor/kor2 kor/kor
grep -vP "Formal" kor/kor > kor/kor2; mv kor/kor2 kor/kor
grep -vP "Informal" kor/kor > kor/kor2; mv kor/kor2 kor/kor
sed "s/CV;/CVB;/" kor/kor > kor/kor2; mv kor/kor2 kor/kor

# Basque
sed "s/PRES/PRS/" eus/eus > eus/eus2; mv eus/eus2 eus/eus
sed "s/PAST/PST/" eus/eus > eus/eus2; mv eus/eus2 eus/eus

# Sorbian
sed "s/:/;/" hsb/hsb > hsb/hsb2; mv hsb/hsb2 hsb/hsb
sed "s/:3:/;3;/" hsb/hsb > hsb/hsb2; mv hsb/hsb2 hsb/hsb

# Slovak
unxz slk/slk.xz
cut -d$'\t' -f 1-3 slk/slk > slk/slk2; mv slk/slk2 slk/slk

# Kazakh
cat kaz/kaz.sm >> kaz/kaz

# Sanskrit
sed "s/PRES/PRS/" san/san > san/san2; mv san/san2 san/san
sed "s/NEU/NEUT/" san/san > san/san2; mv san/san2 san/san
sed "s/{//g; s/}//g" san/san > san/san2; mv san/san2 san/san

# Dutch
echo -e "willen\twil\tV;IND;PRS;2;SG" >> nld/nld
echo -e "zullen\tzal\tV;IND;PRS;2;SG" >> nld/nld

# German
grep -v ADP deu/deu.segmentations > deu/deu2.segmentations; mv deu/deu2.segmentations deu/deu.segmentations

# Kurmanji
sed "s/1;2;3;/1,2,3;/" kmr/kmr > kmr/kmr2; mv kmr/kmr2 kmr/kmr
