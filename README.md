# Link_Prediction_Model with an example

`example.ipynb`に`Link_Prediction_Model`の使用例を記載しています。

詳細な設定は、`models.py`のdocstringをご覧ください。

## ディレクトリ構成図

レポジトリ内部ではなく、親ディレクトリにデータを保持していることに注意。  

```
.  
├── data
│   ├── cora <-自動で作成される
│   └── factset
│         └── processed_data  
│               ├── edge.csv  
│               ├── feature.csv  
│               └── label.csv  
└── link_prediction <-We are here!  
      ├── model.py  
      ├── my_util.py  
      ├── example.ipynb
      └── output <-自動で作成される 
```
