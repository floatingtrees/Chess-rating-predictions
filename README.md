# Chess-rating-predictions


Predicting the ratings of Lichess chess players based on their moves. The best solution has a mean absolute error of around 160 rating points, which amounts to approximately 0.35 standard deviations. 

The data is from [https://database.lichess.org/]([url](https://database.lichess.org/))

The tested models included a transformer, LSTM, and bag-of-words model. The transformer performed the best, followed by the bag-of-words model, with the LSTM trailing significantly behind. 

# How to run:
Download a .pgn.bz2 file, decompress it, and rename the file into a csv. To process data, first run processing.py, followed by time_aligner.py, and finally vectorize.py. This process should take anywhere from 20 minutes to an hour, depending on your computer and the size of the downloaded Lichess file, due to the large size of the files. 
