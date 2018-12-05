# pysoundfinder

Sound localization software for Python

## About

This software is a Python reimplementation of [Sound Finder](https://doi.org/10.1080/09524622.2013.827588) (Wilson et al. 2014), a sound localization program originally written for R and Microsoft Excel. Using the sound’s time delay of arrival (TDOA) at each microphone in an array, PySoundFinder outputs the least-squares solution. The program also returns an “error value” which is correlated with localization accuracy (Wilson et al. 2014).

## How to use

This software requires two `.csv` inputs, one containing recorder positions in meters, and one containing sound TDOAs and temperatures at time of sound arrival. Currently, the sound TDOAs must be obtained through another program, e.g., the waveform cross-correlation function of Raven Pro (Cornell Lab of Ornithology Bioacoustics Research Program, Ithaca, NY, USA).

The position `.csv` should be of format (`z` values optional)

```
recorder,x,y,z
r1,0,0,3
r2,0,30,4
r3,30,0,6
r4,30,30,3
```

where positions are in meters. These positions can either be relative to each other (as above), or absolute UTM coordinates.

The TDOA `.csv` should be of format

```
idx,r1,r2,r3,r4,temp
0,0.07499852061526328,0.017770076718419108,0.1058863479116412,0.07683043801883181,20.0
```

where `idx` is a name or index for the sound to be localized, `r1`...`r4` are the times of arrival of the sound at each recorder, and `temp` is the ambient temperature **in Celsius** at the time of the sound’s arrival. 

Both `.csv` files should have the exact same recorder names in the exact order; otherwise, PySF will throw an error and exit.


Following is a simple example using the above files as `positions.csv` and `sounds.csv`:

```
import pysoundfinder as pysf
pysf.all_sounds('positions.csv', 'sounds.csv')
```

The above will give a plot of the recorders and sounds as output.

## Details
The original [Sound Finder](https://doi.org/10.1080/09524622.2013.827588) described by Wilson et al. (2014) is an implementation of the mathematically equivalent GPS position/time estimation method of Bancroft et al. (1985), more readably described by Halverson (2010).

This reimplementation is modeled after the R version of Sound Finder, but it differs in several ways. 
* The R implementation uses the [`qr` function](https://stat.ethz.ch/R-manual/R-devel/library/base/html/qr.html) to return the **compact form** of the QR decomposition of matrix `B`. No equivalent Python function is apparent, so instead the equations are solved directly through matrix inversion. 
* Because a quadratic equation is solved, two potential solutions are generated. Unlike the original Sound Finder, the returned location is the solution with the lower error value, instead of the lower sum of squares discrepancy; tests found that the error value reliably returned the correct solution in both cases. 
* PySF does not center the TDOA inputs around their mean (yet). 
* The usage and file format of inputs are different (see above).


## Works cited
Bancroft, Stephen. “An Algebraic Solution of the GPS Equations.” *IEEE Transactions on Aerospace and Electronic Systems* AES-21, no. 1 (January 1985): 56–59. https://doi.org/10.1109/TAES.1985.310538.

Halverson, T. “Global Positioning Systems,” October 2002. http://web.archive.org/web/20110719232148/http://www.macalester.edu/~halverson/math36/GPS.pdf.

Wilson, David R., Matthew Battiston, John Brzustowski, and Daniel J. Mennill. “Sound Finder: A New Software Approach for Localizing Animals Recorded with a Microphone Array.” *Bioacoustics* 23, no. 2 (May 4, 2014): 99–112. https://doi.org/10.1080/09524622.2013.827588.

Cornell Bioacoustics Research Program. *Raven Pro: Interactive Sound Analysis Software* (version 1.5). Ithaca, NY: The Cornell Lab of Ornithology, 2014. http://www.birds.cornell.edu/raven.

