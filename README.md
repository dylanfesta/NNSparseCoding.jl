# NNSparseCoding
<p xmlns:dct="http://purl.org/dc/terms/">
  <a rel="license"
     href="http://creativecommons.org/publicdomain/zero/1.0/">
    <img src="https://licensebuttons.net/p/zero/1.0/88x31.png" style="border-style: none;" alt="CC0" />
  </a>
</p>

:warning: **WORK IN PROGRESS**

Non-Negative sparse coding for natural images.

This project aims to replicate the results in
> P. O. Hoyer, “Modeling receptive fields with non-negative sparse coding,” Neurocomputing, vol. 52–54, pp. 547–552, Jun. 2003, doi: 10.1016/S0925-2312(02)00782-8.

And
> P. O. Hoyer and A. Hyvarinen, “Interpreting neural response variability as Monte Carlo sampling of the posterior,” in Advances in neural information processing systems, 2003, pp. 293–300. <http://papers.nips.cc/paper/2152-interpreting-neural-response-variability-as-monte-carlo-sampling-of-the-posterior.pdf>

The algorithm for sparse, non-negative decomposition is described in
> P. O. Hoyer, “Non-negative sparse coding,” in Proceedings of the 12th IEEE Workshop on Neural Networks for Signal Processing, 2002, pp. 557–565, doi: 10.1109/NNSP.2002.1030067.

For the sampling procedure, I will rely on a probabilistc programming toolbox rather than implementing from scratch the Monte Carlo proposed in the NIPS 2002 paper.


_____
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
