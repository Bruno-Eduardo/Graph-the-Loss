#run this commanda with:
#    nano data.txt2; mv data.txt2 data.txt;./cmd
cat data.txt | grep loss | sed 's/acc.*//' | sed 's/.*loss\: //' | sed 's/ - .*//' > data.csv;
cat data.txt | grep val_loss | sed 's/.*val_loss\: //' | sed 's/ .*//' > valLoss.csv
