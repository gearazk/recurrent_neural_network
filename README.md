# Recurrent Reural Network

Sample RNN code with 3 basic layers

## Overview

This is the code for [this](https://youtu.be/BwmddtPFWtA) video on Youtube by Siraj Raval as part of The Math of Intelligence series. It's a simple numpy implementation of a recurrent network that can read in a sequence of words and then generate words in that style. We train it on Franz Kafka here. 

## Dependencies

* numpy

Install missing ependencies with [pip](https://pip.pypa.io/en/stable/). 

## Usage

Type `jupyter notebook` in terminal when in the main directory and it will pop up in your browser.

Install [jupyter](http://jupyter.readthedocs.io/en/latest/install.html) if you haven't yet 

Run
```commandline
python ./RNN.py
```

Edit the  `input.txt` file to feed different content to the Model

Sample output

```

----
iter 1000, loss: 85.255375
----
  cthlanw bo vse pooG thacrogad e. iw matt reuchislinls arpwtortrehpmort henet thiced ehkjd fhe orijt the wiftf bots iv do weo mranat ko? "i gfbt fid the Ohe boms tbty ehe has hasteharmphinhu moerenL a 
----
iter 2000, loss: 69.093164
----
 methom inef le cthe heth he Norut wout hitgero't onst toueud ithe lseins ohosatheoga Ine che thids
palild his ateroworde o thamtll ase has usens boly jlun he thegke he woostam, mor ohe shcor enn wonse 
----

...

----
iter 99000, loss: 43.503157
----
 on plroneess in requlctidutirghtire! CTmected a fate, was and in tteip a reforutewn
tist; out afmaiges oll walich ter ave fopight whoog to not of enolried Sas ed om apperent, by the g.

  OD juck EOV
 
----
iter 100000, loss: 43.111022
----
 gh sisted semss and you" handmild to work stoble's praint hard ofte to get in Gregor of all knock whainbe some, you not much while a anyow chere distes's byther tht ot to he. Oreancaboftund to gear" n 
----

```

## Credits

The credits for this code go to [Andrej Karpathy](https://gist.github.com/karpathy/d4dee566867f8291f086). My favorite DL researcher <3 I've merely created a wrapper to get people started. 

And the explanation from this guy [Siraj Raval](https://www.youtube.com/watch?v=BwmddtPFWtA).  
