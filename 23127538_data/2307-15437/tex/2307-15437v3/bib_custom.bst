ENTRY
{ address
  author
  title
  journal
  year
  volume
 }
  
INCLUDE
{ "article" }
FUNCTION {format.names}
{ 's :=
  #3  'et.al. *  % この行を変更して、3人目以降に 'et al.' を追加
}

FUNCTION {article}
{ % article タイプの文献の書式を定義
  author "title" journal year volume pages
}


