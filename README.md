## Inferring Countries from Publication Affiliations
Given an affiliation string, infer the country of the affiliation.
For example, given the affiliation string 
`Beth Israel Deaconess Medical Center`, the inferred country is `US`.

## Usage
**STEP 1:** 
Clone this repo and unzip the training data file (which is 160MB after unzipped).
The training data includes unique affiliation strings from 249 countries, derived from 
[ORCID public data](https://info.orcid.org/documentation/integration-guide/working-with-bulk-data/).
```
git clone https://github.com/junwang4/affiliation-to-country-inference
cd affiliation-to-country-inference
gunzip data/train_orcid.csv.gz
``` 
(See the table at the end for the country distribution.)

**STEP 2: Train a LinearSVC-based text classification model** 

Edit the `settings.py` file to specify the directory where your trained models will be stored and the countries you are interested in.

Keep in mind that if you use the default setting of 50 countries in settings.py, the trained model will require
about 1.5GB of storage. And if you use all the 249 countries, the trained model will require about 10 to 20GB of storage.

For example, if your model directory is `/tmp/models`, you can run the following command:
```
python3 aff2country.py --task=train --input=data/train_orcid.csv
```

**STEP 3: Apply the trained model** 

After training, you can use the model (e.g., located in `/tmp/models`) to infer the country from an affiliation string.
```
python aff2country.py --task=predict --input=data/sample1000.csv
```

Suppose you have a file `data/sample1000.csv` with the following content:
```
text
"Department of Medicine, University of klahoma Health Sciences Center, Oklahoma City."
"Department of Microbiology, Dr. Ziauddin University Hospital, Karachi. afridi03@hotmail.com"
"Macedonian Academy of Sciences and Arts, Skopje, R. Macedonia."
...
```

The prediction will be saved to `data/sample.pred.csv`.
(Note that the country code `--` means the inferred country is not in the list of the 50 countries specified in the `settings.py`).
```
text,confidence,winner
"Department of Medicine, University of klahoma Health Sciences Center, Oklahoma City.",0.982,US
"Department of Microbiology, Dr. Ziauddin University Hospital, Karachi. afridi03@hotmail.com",0.972,PK
"Macedonian Academy of Sciences and Arts, Skopje, R. Macedonia.",0.982,--
```



### Distribution of Countries in the Training Data ###
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BR</td>
      <td>277755</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>245720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IN</td>
      <td>175932</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ES</td>
      <td>116421</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GB</td>
      <td>96894</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CN</td>
      <td>93299</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CO</td>
      <td>77519</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RU</td>
      <td>77045</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MX</td>
      <td>65040</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PT</td>
      <td>64778</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FR</td>
      <td>64596</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DE</td>
      <td>62441</td>
    </tr>
    <tr>
      <th>12</th>
      <td>IT</td>
      <td>53488</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PE</td>
      <td>52904</td>
    </tr>
    <tr>
      <th>14</th>
      <td>AU</td>
      <td>50174</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TR</td>
      <td>49419</td>
    </tr>
    <tr>
      <th>16</th>
      <td>UA</td>
      <td>48798</td>
    </tr>
    <tr>
      <th>17</th>
      <td>JP</td>
      <td>46330</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PL</td>
      <td>41475</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ID</td>
      <td>39155</td>
    </tr>
    <tr>
      <th>20</th>
      <td>IR</td>
      <td>36452</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CA</td>
      <td>33323</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ZA</td>
      <td>28256</td>
    </tr>
    <tr>
      <th>23</th>
      <td>KR</td>
      <td>26531</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NG</td>
      <td>26484</td>
    </tr>
    <tr>
      <th>25</th>
      <td>AR</td>
      <td>25068</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PK</td>
      <td>23971</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CH</td>
      <td>23170</td>
    </tr>
    <tr>
      <th>28</th>
      <td>BE</td>
      <td>22070</td>
    </tr>
    <tr>
      <th>29</th>
      <td>EG</td>
      <td>21961</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NL</td>
      <td>20674</td>
    </tr>
    <tr>
      <th>31</th>
      <td>CL</td>
      <td>20353</td>
    </tr>
    <tr>
      <th>32</th>
      <td>EC</td>
      <td>20199</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MY</td>
      <td>19525</td>
    </tr>
    <tr>
      <th>34</th>
      <td>SE</td>
      <td>18850</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PH</td>
      <td>17597</td>
    </tr>
    <tr>
      <th>36</th>
      <td>CU</td>
      <td>15944</td>
    </tr>
    <tr>
      <th>37</th>
      <td>IQ</td>
      <td>15280</td>
    </tr>
    <tr>
      <th>38</th>
      <td>BD</td>
      <td>14696</td>
    </tr>
    <tr>
      <th>39</th>
      <td>DK</td>
      <td>14611</td>
    </tr>
    <tr>
      <th>40</th>
      <td>AT</td>
      <td>14145</td>
    </tr>
    <tr>
      <th>41</th>
      <td>GR</td>
      <td>13590</td>
    </tr>
    <tr>
      <th>42</th>
      <td>CZ</td>
      <td>13532</td>
    </tr>
    <tr>
      <th>43</th>
      <td>RO</td>
      <td>13461</td>
    </tr>
    <tr>
      <th>44</th>
      <td>VE</td>
      <td>11869</td>
    </tr>
    <tr>
      <th>45</th>
      <td>TW</td>
      <td>10967</td>
    </tr>
    <tr>
      <th>46</th>
      <td>IE</td>
      <td>10895</td>
    </tr>
    <tr>
      <th>47</th>
      <td>FI</td>
      <td>10498</td>
    </tr>
    <tr>
      <th>48</th>
      <td>SA</td>
      <td>10099</td>
    </tr>
    <tr>
      <th>49</th>
      <td>VN</td>
      <td>10042</td>
    </tr>
    <tr>
      <th>50</th>
      <td>KE</td>
      <td>9794</td>
    </tr>
    <tr>
      <th>51</th>
      <td>RS</td>
      <td>9477</td>
    </tr>
    <tr>
      <th>52</th>
      <td>NO</td>
      <td>9330</td>
    </tr>
    <tr>
      <th>53</th>
      <td>ET</td>
      <td>8721</td>
    </tr>
    <tr>
      <th>54</th>
      <td>DZ</td>
      <td>8320</td>
    </tr>
    <tr>
      <th>55</th>
      <td>NP</td>
      <td>8309</td>
    </tr>
    <tr>
      <th>56</th>
      <td>TH</td>
      <td>8202</td>
    </tr>
    <tr>
      <th>57</th>
      <td>HU</td>
      <td>8133</td>
    </tr>
    <tr>
      <th>58</th>
      <td>HR</td>
      <td>7929</td>
    </tr>
    <tr>
      <th>59</th>
      <td>KZ</td>
      <td>7833</td>
    </tr>
    <tr>
      <th>60</th>
      <td>NZ</td>
      <td>7822</td>
    </tr>
    <tr>
      <th>61</th>
      <td>MA</td>
      <td>7138</td>
    </tr>
    <tr>
      <th>62</th>
      <td>GH</td>
      <td>6663</td>
    </tr>
    <tr>
      <th>63</th>
      <td>LK</td>
      <td>6527</td>
    </tr>
    <tr>
      <th>64</th>
      <td>IL</td>
      <td>6332</td>
    </tr>
    <tr>
      <th>65</th>
      <td>BG</td>
      <td>6315</td>
    </tr>
    <tr>
      <th>66</th>
      <td>HK</td>
      <td>6307</td>
    </tr>
    <tr>
      <th>67</th>
      <td>TN</td>
      <td>6200</td>
    </tr>
    <tr>
      <th>68</th>
      <td>SG</td>
      <td>5926</td>
    </tr>
    <tr>
      <th>69</th>
      <td>AE</td>
      <td>5864</td>
    </tr>
    <tr>
      <th>70</th>
      <td>SK</td>
      <td>4960</td>
    </tr>
    <tr>
      <th>71</th>
      <td>JO</td>
      <td>4733</td>
    </tr>
    <tr>
      <th>72</th>
      <td>UZ</td>
      <td>4619</td>
    </tr>
    <tr>
      <th>73</th>
      <td>CR</td>
      <td>4533</td>
    </tr>
    <tr>
      <th>74</th>
      <td>UG</td>
      <td>4507</td>
    </tr>
    <tr>
      <th>75</th>
      <td>CM</td>
      <td>4230</td>
    </tr>
    <tr>
      <th>76</th>
      <td>UY</td>
      <td>4129</td>
    </tr>
    <tr>
      <th>77</th>
      <td>TZ</td>
      <td>3977</td>
    </tr>
    <tr>
      <th>78</th>
      <td>BY</td>
      <td>3972</td>
    </tr>
    <tr>
      <th>79</th>
      <td>LB</td>
      <td>3638</td>
    </tr>
    <tr>
      <th>80</th>
      <td>PY</td>
      <td>3481</td>
    </tr>
    <tr>
      <th>81</th>
      <td>SI</td>
      <td>3457</td>
    </tr>
    <tr>
      <th>82</th>
      <td>SD</td>
      <td>3395</td>
    </tr>
    <tr>
      <th>83</th>
      <td>BO</td>
      <td>3331</td>
    </tr>
    <tr>
      <th>84</th>
      <td>MZ</td>
      <td>3225</td>
    </tr>
    <tr>
      <th>85</th>
      <td>LT</td>
      <td>3224</td>
    </tr>
    <tr>
      <th>86</th>
      <td>DO</td>
      <td>3171</td>
    </tr>
    <tr>
      <th>87</th>
      <td>ZW</td>
      <td>3161</td>
    </tr>
    <tr>
      <th>88</th>
      <td>BA</td>
      <td>3102</td>
    </tr>
    <tr>
      <th>89</th>
      <td>PA</td>
      <td>2918</td>
    </tr>
    <tr>
      <th>90</th>
      <td>CY</td>
      <td>2716</td>
    </tr>
    <tr>
      <th>91</th>
      <td>AZ</td>
      <td>2659</td>
    </tr>
    <tr>
      <th>92</th>
      <td>LV</td>
      <td>2655</td>
    </tr>
    <tr>
      <th>93</th>
      <td>GE</td>
      <td>2599</td>
    </tr>
    <tr>
      <th>94</th>
      <td>RW</td>
      <td>2556</td>
    </tr>
    <tr>
      <th>95</th>
      <td>PR</td>
      <td>2530</td>
    </tr>
    <tr>
      <th>96</th>
      <td>AL</td>
      <td>2525</td>
    </tr>
    <tr>
      <th>97</th>
      <td>AO</td>
      <td>2507</td>
    </tr>
    <tr>
      <th>98</th>
      <td>MK</td>
      <td>2492</td>
    </tr>
    <tr>
      <th>99</th>
      <td>QA</td>
      <td>2488</td>
    </tr>
    <tr>
      <th>100</th>
      <td>PS</td>
      <td>2480</td>
    </tr>
    <tr>
      <th>101</th>
      <td>AM</td>
      <td>2322</td>
    </tr>
    <tr>
      <th>102</th>
      <td>EE</td>
      <td>2318</td>
    </tr>
    <tr>
      <th>103</th>
      <td>SY</td>
      <td>2286</td>
    </tr>
    <tr>
      <th>104</th>
      <td>LY</td>
      <td>2264</td>
    </tr>
    <tr>
      <th>105</th>
      <td>OM</td>
      <td>2247</td>
    </tr>
    <tr>
      <th>106</th>
      <td>CD</td>
      <td>2245</td>
    </tr>
    <tr>
      <th>107</th>
      <td>ZM</td>
      <td>2234</td>
    </tr>
    <tr>
      <th>108</th>
      <td>MW</td>
      <td>2226</td>
    </tr>
    <tr>
      <th>109</th>
      <td>MD</td>
      <td>2173</td>
    </tr>
    <tr>
      <th>110</th>
      <td>MN</td>
      <td>2154</td>
    </tr>
    <tr>
      <th>111</th>
      <td>MM</td>
      <td>2143</td>
    </tr>
    <tr>
      <th>112</th>
      <td>NI</td>
      <td>2085</td>
    </tr>
    <tr>
      <th>113</th>
      <td>IS</td>
      <td>2067</td>
    </tr>
    <tr>
      <th>114</th>
      <td>HN</td>
      <td>2054</td>
    </tr>
    <tr>
      <th>115</th>
      <td>AF</td>
      <td>2039</td>
    </tr>
    <tr>
      <th>116</th>
      <td>YE</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>117</th>
      <td>LU</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>118</th>
      <td>SV</td>
      <td>1958</td>
    </tr>
    <tr>
      <th>119</th>
      <td>KG</td>
      <td>1877</td>
    </tr>
    <tr>
      <th>120</th>
      <td>SN</td>
      <td>1846</td>
    </tr>
    <tr>
      <th>121</th>
      <td>BJ</td>
      <td>1805</td>
    </tr>
    <tr>
      <th>122</th>
      <td>CI</td>
      <td>1702</td>
    </tr>
    <tr>
      <th>123</th>
      <td>BF</td>
      <td>1697</td>
    </tr>
    <tr>
      <th>124</th>
      <td>MT</td>
      <td>1614</td>
    </tr>
    <tr>
      <th>125</th>
      <td>GT</td>
      <td>1584</td>
    </tr>
    <tr>
      <th>126</th>
      <td>BW</td>
      <td>1392</td>
    </tr>
    <tr>
      <th>127</th>
      <td>KW</td>
      <td>1358</td>
    </tr>
    <tr>
      <th>128</th>
      <td>FJ</td>
      <td>1204</td>
    </tr>
    <tr>
      <th>129</th>
      <td>KH</td>
      <td>1194</td>
    </tr>
    <tr>
      <th>130</th>
      <td>MG</td>
      <td>1175</td>
    </tr>
    <tr>
      <th>131</th>
      <td>BT</td>
      <td>1152</td>
    </tr>
    <tr>
      <th>132</th>
      <td>SO</td>
      <td>1123</td>
    </tr>
    <tr>
      <th>133</th>
      <td>BH</td>
      <td>1099</td>
    </tr>
    <tr>
      <th>134</th>
      <td>ML</td>
      <td>1057</td>
    </tr>
    <tr>
      <th>135</th>
      <td>CV</td>
      <td>982</td>
    </tr>
    <tr>
      <th>136</th>
      <td>TT</td>
      <td>946</td>
    </tr>
    <tr>
      <th>137</th>
      <td>SZ</td>
      <td>902</td>
    </tr>
    <tr>
      <th>138</th>
      <td>JM</td>
      <td>821</td>
    </tr>
    <tr>
      <th>139</th>
      <td>TJ</td>
      <td>794</td>
    </tr>
    <tr>
      <th>140</th>
      <td>MO</td>
      <td>766</td>
    </tr>
    <tr>
      <th>141</th>
      <td>SL</td>
      <td>729</td>
    </tr>
    <tr>
      <th>142</th>
      <td>LR</td>
      <td>703</td>
    </tr>
    <tr>
      <th>143</th>
      <td>HT</td>
      <td>691</td>
    </tr>
    <tr>
      <th>144</th>
      <td>ME</td>
      <td>653</td>
    </tr>
    <tr>
      <th>145</th>
      <td>MU</td>
      <td>651</td>
    </tr>
    <tr>
      <th>146</th>
      <td>TG</td>
      <td>625</td>
    </tr>
    <tr>
      <th>147</th>
      <td>PG</td>
      <td>563</td>
    </tr>
    <tr>
      <th>148</th>
      <td>ER</td>
      <td>525</td>
    </tr>
    <tr>
      <th>149</th>
      <td>GM</td>
      <td>524</td>
    </tr>
    <tr>
      <th>150</th>
      <td>NE</td>
      <td>496</td>
    </tr>
    <tr>
      <th>151</th>
      <td>TL</td>
      <td>482</td>
    </tr>
    <tr>
      <th>152</th>
      <td>GN</td>
      <td>460</td>
    </tr>
    <tr>
      <th>153</th>
      <td>CG</td>
      <td>450</td>
    </tr>
    <tr>
      <th>154</th>
      <td>BI</td>
      <td>408</td>
    </tr>
    <tr>
      <th>155</th>
      <td>BN</td>
      <td>403</td>
    </tr>
    <tr>
      <th>156</th>
      <td>SS</td>
      <td>402</td>
    </tr>
    <tr>
      <th>157</th>
      <td>LA</td>
      <td>399</td>
    </tr>
    <tr>
      <th>158</th>
      <td>GA</td>
      <td>397</td>
    </tr>
    <tr>
      <th>159</th>
      <td>LS</td>
      <td>393</td>
    </tr>
    <tr>
      <th>160</th>
      <td>BB</td>
      <td>372</td>
    </tr>
    <tr>
      <th>161</th>
      <td>RE</td>
      <td>348</td>
    </tr>
    <tr>
      <th>162</th>
      <td>MV</td>
      <td>339</td>
    </tr>
    <tr>
      <th>163</th>
      <td>BS</td>
      <td>336</td>
    </tr>
    <tr>
      <th>164</th>
      <td>TD</td>
      <td>333</td>
    </tr>
    <tr>
      <th>165</th>
      <td>NC</td>
      <td>272</td>
    </tr>
    <tr>
      <th>166</th>
      <td>GD</td>
      <td>258</td>
    </tr>
    <tr>
      <th>167</th>
      <td>GW</td>
      <td>255</td>
    </tr>
    <tr>
      <th>168</th>
      <td>UM</td>
      <td>250</td>
    </tr>
    <tr>
      <th>169</th>
      <td>VI</td>
      <td>237</td>
    </tr>
    <tr>
      <th>170</th>
      <td>LI</td>
      <td>234</td>
    </tr>
    <tr>
      <th>171</th>
      <td>PF</td>
      <td>219</td>
    </tr>
    <tr>
      <th>172</th>
      <td>VA</td>
      <td>216</td>
    </tr>
    <tr>
      <th>173</th>
      <td>GY</td>
      <td>214</td>
    </tr>
    <tr>
      <th>174</th>
      <td>GF</td>
      <td>205</td>
    </tr>
    <tr>
      <th>175</th>
      <td>AS</td>
      <td>199</td>
    </tr>
    <tr>
      <th>176</th>
      <td>KP</td>
      <td>189</td>
    </tr>
    <tr>
      <th>177</th>
      <td>MR</td>
      <td>189</td>
    </tr>
    <tr>
      <th>178</th>
      <td>GP</td>
      <td>185</td>
    </tr>
    <tr>
      <th>179</th>
      <td>AD</td>
      <td>183</td>
    </tr>
    <tr>
      <th>180</th>
      <td>SC</td>
      <td>173</td>
    </tr>
    <tr>
      <th>181</th>
      <td>BZ</td>
      <td>173</td>
    </tr>
    <tr>
      <th>182</th>
      <td>ST</td>
      <td>172</td>
    </tr>
    <tr>
      <th>183</th>
      <td>SR</td>
      <td>169</td>
    </tr>
    <tr>
      <th>184</th>
      <td>VU</td>
      <td>166</td>
    </tr>
    <tr>
      <th>185</th>
      <td>CF</td>
      <td>165</td>
    </tr>
    <tr>
      <th>186</th>
      <td>CW</td>
      <td>154</td>
    </tr>
    <tr>
      <th>187</th>
      <td>FO</td>
      <td>149</td>
    </tr>
    <tr>
      <th>188</th>
      <td>MQ</td>
      <td>149</td>
    </tr>
    <tr>
      <th>189</th>
      <td>TM</td>
      <td>142</td>
    </tr>
    <tr>
      <th>190</th>
      <td>MC</td>
      <td>134</td>
    </tr>
    <tr>
      <th>191</th>
      <td>KN</td>
      <td>132</td>
    </tr>
    <tr>
      <th>192</th>
      <td>KM</td>
      <td>120</td>
    </tr>
    <tr>
      <th>193</th>
      <td>GQ</td>
      <td>115</td>
    </tr>
    <tr>
      <th>194</th>
      <td>WS</td>
      <td>110</td>
    </tr>
    <tr>
      <th>195</th>
      <td>XK</td>
      <td>100</td>
    </tr>
    <tr>
      <th>196</th>
      <td>BQ</td>
      <td>100</td>
    </tr>
    <tr>
      <th>197</th>
      <td>DM</td>
      <td>97</td>
    </tr>
    <tr>
      <th>198</th>
      <td>GL</td>
      <td>92</td>
    </tr>
    <tr>
      <th>199</th>
      <td>SB</td>
      <td>90</td>
    </tr>
    <tr>
      <th>200</th>
      <td>KY</td>
      <td>89</td>
    </tr>
    <tr>
      <th>201</th>
      <td>BM</td>
      <td>88</td>
    </tr>
    <tr>
      <th>202</th>
      <td>IM</td>
      <td>80</td>
    </tr>
    <tr>
      <th>203</th>
      <td>JE</td>
      <td>80</td>
    </tr>
    <tr>
      <th>204</th>
      <td>AQ</td>
      <td>79</td>
    </tr>
    <tr>
      <th>205</th>
      <td>IO</td>
      <td>78</td>
    </tr>
    <tr>
      <th>206</th>
      <td>AG</td>
      <td>72</td>
    </tr>
    <tr>
      <th>207</th>
      <td>GI</td>
      <td>66</td>
    </tr>
    <tr>
      <th>208</th>
      <td>LC</td>
      <td>63</td>
    </tr>
    <tr>
      <th>209</th>
      <td>VC</td>
      <td>62</td>
    </tr>
    <tr>
      <th>210</th>
      <td>AW</td>
      <td>61</td>
    </tr>
    <tr>
      <th>211</th>
      <td>VG</td>
      <td>56</td>
    </tr>
    <tr>
      <th>212</th>
      <td>AX</td>
      <td>54</td>
    </tr>
    <tr>
      <th>213</th>
      <td>GU</td>
      <td>54</td>
    </tr>
    <tr>
      <th>214</th>
      <td>DJ</td>
      <td>53</td>
    </tr>
    <tr>
      <th>215</th>
      <td>KI</td>
      <td>49</td>
    </tr>
    <tr>
      <th>216</th>
      <td>BV</td>
      <td>46</td>
    </tr>
    <tr>
      <th>217</th>
      <td>TO</td>
      <td>46</td>
    </tr>
    <tr>
      <th>218</th>
      <td>CX</td>
      <td>45</td>
    </tr>
    <tr>
      <th>219</th>
      <td>MP</td>
      <td>44</td>
    </tr>
    <tr>
      <th>220</th>
      <td>SM</td>
      <td>43</td>
    </tr>
    <tr>
      <th>221</th>
      <td>SX</td>
      <td>43</td>
    </tr>
    <tr>
      <th>222</th>
      <td>TC</td>
      <td>42</td>
    </tr>
    <tr>
      <th>223</th>
      <td>FM</td>
      <td>41</td>
    </tr>
    <tr>
      <th>224</th>
      <td>TV</td>
      <td>36</td>
    </tr>
    <tr>
      <th>225</th>
      <td>FK</td>
      <td>36</td>
    </tr>
    <tr>
      <th>226</th>
      <td>AI</td>
      <td>35</td>
    </tr>
    <tr>
      <th>227</th>
      <td>PW</td>
      <td>34</td>
    </tr>
    <tr>
      <th>228</th>
      <td>MS</td>
      <td>32</td>
    </tr>
    <tr>
      <th>229</th>
      <td>PN</td>
      <td>30</td>
    </tr>
    <tr>
      <th>230</th>
      <td>CK</td>
      <td>29</td>
    </tr>
    <tr>
      <th>231</th>
      <td>GS</td>
      <td>28</td>
    </tr>
    <tr>
      <th>232</th>
      <td>NR</td>
      <td>27</td>
    </tr>
    <tr>
      <th>233</th>
      <td>YT</td>
      <td>24</td>
    </tr>
    <tr>
      <th>234</th>
      <td>SJ</td>
      <td>22</td>
    </tr>
    <tr>
      <th>235</th>
      <td>NF</td>
      <td>20</td>
    </tr>
    <tr>
      <th>236</th>
      <td>GG</td>
      <td>19</td>
    </tr>
    <tr>
      <th>237</th>
      <td>MF</td>
      <td>18</td>
    </tr>
    <tr>
      <th>238</th>
      <td>BL</td>
      <td>18</td>
    </tr>
    <tr>
      <th>239</th>
      <td>NU</td>
      <td>17</td>
    </tr>
    <tr>
      <th>240</th>
      <td>MH</td>
      <td>17</td>
    </tr>
    <tr>
      <th>241</th>
      <td>CC</td>
      <td>14</td>
    </tr>
    <tr>
      <th>242</th>
      <td>TF</td>
      <td>14</td>
    </tr>
    <tr>
      <th>243</th>
      <td>TK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>244</th>
      <td>WF</td>
      <td>14</td>
    </tr>
    <tr>
      <th>245</th>
      <td>EH</td>
      <td>13</td>
    </tr>
    <tr>
      <th>246</th>
      <td>SH</td>
      <td>12</td>
    </tr>
    <tr>
      <th>247</th>
      <td>HM</td>
      <td>8</td>
    </tr>
    <tr>
      <th>248</th>
      <td>PM</td>
      <td>6</td>
    </tr>
  </tbody>
</table>