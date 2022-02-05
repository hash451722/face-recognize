# Face Recognize

## Environment setting

confirmed in 2022-1-29

```
conda create -n face python=3.8
conda activate face
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
```

```
Python 3.8.12
OpenCV 4.5.5
Numpy 1.22.0
Matplotlib 3.5.1
```


## Detect

https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet


## Recognition

https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface



## Face idx
<table>
  <tr>
    <th>0</th>
    <th>1</th>
    <th>2</th>
    <th>3</th>
    <th>4</th>
    <th>5</th>
    <th>6</th>
    <th>7</th>
    <th>8</th>
    <th>9</th>
    <th>10</th>
    <th>11</th>
    <th>12</th>
    <th>13</th>
    <th>14</th>
  </tr>
  <tr>
    <td colspan="4">Bounding box</td>
    <td colspan="10">Landmark</td>
    <td>Score</td>
  </tr>
  <tr>
    <td colspan=2>Top left corner</td>
    <td rowspan=2>Width</td>
    <td rowspan=2>Height</td>
    <td colspan=2>Right eye</td>
    <td colspan=2>Left eye</td>
    <td colspan=2>Nose</td>
    <td colspan=2>Right mouth corner</td>
    <td colspan=2>Left mouth corner</td>
    <td rowspan=2>Score</td>
  </tr>
  <ty>
    <td>x-coords</td>
    <td>y-coords</td>
    <td>x-coords</td>
    <td>y-coords</td>
    <td>x-coords</td>
    <td>y-coords</td>
    <td>x-coords</td>
    <td>y-coords</td>
    <td>x-coords</td>
    <td>y-coords</td>
    <td>x-coords</td>
    <td>y-coords</td>
  </tr>
</table>
