## Obfuscating Keystroke Time Intervals to Avoid Identification and Impersonation

**Abstract:** There are numerous opportunities for adversaries to observe user behavior remotely on the web. Additionally, keystroke biometric algorithms have advanced to the point where user identification and soft biometric trait recognition rates are commercially viable. This presents a privacy concern because masking spatial information, such as IP address, is not sufficient as users become more identifiable by their behavior. In this work, the well-known Chaum mix is generalized to a scenario in which users are separated by both space and time with the goal of preventing an observing adversary from identifying or impersonating the user. The criteria of a behavior obfuscation strategy are defined and two strategies are introduced for obfuscating typing behavior. Experimental results are obtained using publicly available keystroke data for three different types of input, including short fixed-text, long fixed-text, and long free-text. Identification accuracy is reduced by 20% with a 25 ms random keystroke delay not noticeable to the user.

The full paper can be downloaded [here](http://www.vmonaco.com/publications/Obfuscating%20Keystroke%20Time%20Intervals%20to%20Avoid%20Identification%20and%20Impersonation.pdf?attredirects=0).

#### Citation

    @inproceedings{monaco2016obfuscating,
      title={Obfuscating Keystroke Time Intervals to Avoid Identification and Impersonation},
      author={Monaco, John V and Tappert, Charles C},
      booktitle={The 9th IAPR International Conference on Biometrics (ICB)},
      year={2016},
      organization={IEEE}
    }

#### Steps to reproduce the experiments
 
Run the main script to reproduce all experiments.

`python main.py [seed]`

This will download and preprocess all datasets, followed by obfuscation, feature extraction, and classification. If any of the datasets fails to download, they must be manually placed in the data/download folder. In particular, the Villani dataset requires approved access.

