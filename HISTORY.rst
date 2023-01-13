=======
History
=======

1.3.1 (2023-01-13)
------------------

Changed
********
* Added support for Python 3.11.
* Updated required versions for numpy and scipy.

Fixed
******
* Fixed bug where setup.py would install a directory named tests into site-packages folder (thanks to `Bipin Kumar <https://github.com/kbipinkumar>`)

New Contributors
*****************
* `Bipin Kumar`_

1.3.0 (2022-12-27)
------------------

Added
******
* Added sharpened_cosine_distance().

Changed
********
* Added support for Python 3.8-3.10, and relinquished support for Python <=3.7.
* Updated versions of requirements and developer requirements.

1.2.0 (2020-06-21)
------------------

Added
******
* Updated API to make imports easier (for example: 'from pairwisedist import jackknife_distance' instead of 'from pairwisedist.pairwisedist import jackknife_distance').
* Added pearson_distance() and spearman_distance().

1.1.0 (2020-06-21)
------------------

Added
******
* Added jackknife_distance().


1.0.0 (2020-06-18)
------------------

* First release on PyPI.

Added
******
* Added ys1_distance() and yr1_distance().
