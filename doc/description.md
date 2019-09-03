# 类别描述

```c++
/*大致分为几大类：
  1.NON-NEOPLASTIC FINDINGS； 
  2. 微生物(ORGANISMS); 
  3. OTHER(主要是针对大于45岁的子宫内膜细胞，Specify if “negative for squamous intraepithelial lesion”)； 
  4. 上皮细胞异常； 
  5. 其他恶性肿瘤  
  等
  上皮细胞异常又分为鳞状细胞和腺细胞两大分支*/
/**<cervical cell type*/
enum cctype {
	CELL_BACKGROUND = 0,     //背景类(难负样本Hard negative)
	CELL_ASCUS = 1,          //非典型鳞状细胞，意义不明确(Atypical squamous cells of undetermined significance )
	CELL_ASCH = 2,           //非典型鳞状细胞，不能排除高级别病变(Atypical squamous cells, cannot exclude HSIL (ASC-H))
	CELL_LSIL = 3,           //低级别鳞状上皮内病变(Low-grade squamous intraepithelial lesion (LSIL), 包含HPV/mild dysplasia/CIN 1)
	CELL_HSIL = 4,           //高级别鳞状上皮内病变（High-grade squamous intraepithelial lesion， 包含moderate and severe dysplasia, CIS; CIN 2 and CIN 3）
	CELL_SCC = 5,            //鳞状细胞癌（Squamous cell carcinoma）   
	CELL_AGC = 6,	         //非典型性腺细胞 （Atypical glandular cell）注：这里包括AGC-NOS(not otherwise specified颈内非典型性腺细胞、宫内膜非典型性腺细胞和非典型性腺细胞）和 AGC-FN (颈内非典型性腺细胞favor neoplastic, 非典型性腺细胞favor neoplastic)FN可类比ASCUS-H 
	CELL_AIS = 7,            //颈内原位腺癌AIS(Endocervical adenocarcinoma in situ, 可类比于鳞分支的HSIL)
	CELL_ADENOCARCINOMA = 8, //腺癌 (包括颈内、宫内膜、颈外和NOS)
	CELL_EM = 9,             //子宫内膜细胞(Endometrial cells)
	CELL_VAGINALIS = 10,     //滴虫  (Trichomonas vaginalis)
	CELL_FLORA = 11,         //念珠菌 (Fungal organisms morphologically consistent with Candida spp)
	CELL_DYSBACTERIOSIS = 12,//菌群失调提示细菌性阴道病 (Shift in flora suggestive of bacterial vaginosis)
	CELL_HERPES = 13,        //单纯疱疹病毒(Cellular changes consistent with herpes simplex virus)
	CELL_ACTINOMYCES = 14,   //放线菌(Bacteria morphologically consistent with Actinomyces spp)
	CELL_OMN = 15,           //其它恶性肿瘤 (OTHER MALIGNANT NEOPLASMS)
	CELL_EC  = 16            //颈管细胞
};

```

类别

- 正常细胞
    - em 子宫内膜细胞
    - ec 颈管细胞

- 病变细胞
    - ascus 非典型鳞状细胞，意义不明确 (大细胞，可能病变)
    - lsil 低级别鳞状上皮内病变 (大细胞，低级病变)

    - asch 非典型鳞状细胞，不能排除高级别病变 (小细胞，可能高级病变)
    - hsil 高级别鳞状上皮内病变 (小细胞，高级病变)
    
    - agc 非典型性腺细胞
    
    - dysbacteriosis 菌群失调提示细菌性阴道病
    
    - omn 其它恶性肿瘤

- 癌细胞
    - scc 鳞状细胞癌
    - ais 颈内原位腺癌AIS
    - adenocarcinoma 腺癌
    
- 微生物
    - vaginalis 滴虫
    - flora 念珠菌
    - actinomyces 放线菌
    - herpes 单纯疱疹病毒

```python
categories = [
    "adenocarcinoma",
    "agc",
    "asch",
    "ascus",
    "dysbacteriosis",
    "hsil",
    "lsil",
    "monilia",
    "normal",
    "vaginalis"
]
```
