# Path Loss Parameter Estimation

Specification:

The provided data set is taken from experiments conducted in the fourth floor of RTH. It contains received signal strengths from 802.11 devices located at various fixed locations. Assume that the signal strength decays according to the simplified path-loss model with log-normal fading. Specifically,

  Pr = Pt * K * [d0/d]^Eta * exp(sigma* N / 10lge)

where N is a random variable with the standard normal distribution.
To Do: Use the given dataset to estimate the following model parameters:
• The constant K in dB.
• The path loss exponent, η. 
• The standard deviation, σ.

Data Set:

The received signal powers from 12 experiments are given. In each experiment, 8 receiver devices are placed at the coordinates given in receiverXY.csv. For example, the location of the first device may be read from the first row. The coordinates of the transmitter devices for each experiment is given in transmitterXY.csv. All coordinate values are in units of meters. For each experiment, the received signal power at each device is available in the files wifiExp7.csv, wifiExp8.csv, ..., wifiExp18.csv. Note that the first row of transmitterXY.csv contains the coordinates corresponding to wifiExp7.csv and so on.
Each experiment file contains 9 columns. The first column corresponds to a time stamp that may be ignored. The remaining columns contain the signal powers from the 8 devices in the experiment. For example, column 2 contains values for the first device, column 3 has values for the second device and so on. The signal powers are in units of dBm and their signs have been flipped. In other words, a value of 59.0 in the file should be understood as a received signal power of −59.0 dBm. On occasion, the transmitted packet is lost and consequently the receiver detects only noise. This is indicated in the data files with a received signal power of −500.0 dBm. Ignore these values while estimating the required parameters.
The transmit power in all experiments is −27 dBm and d0 = 1m.
