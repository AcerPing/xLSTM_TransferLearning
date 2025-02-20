#!/bin/bash

cd "$(dirname "$0")" # 獲取腳本所在目錄並切換到該目錄
echo "Current Working Directory: $(pwd)" # 確認當前執行目錄

for dname in `ls source/`; do # 處理 source/ 目錄下的子目錄。
    echo "Processing directory: source/${dname}"
	mkdir -p "../dataset/source/${dname}"  # 確保目標目錄存在
	echo "Copying .pkl files from ./source/${dname} to ../dataset/source/${dname}/"
	cp "./source/${dname}/"*.pkl "../dataset/source/${dname}/" # 使用 cp 複製檔案，而不是 mv
done

for dname in `ls target/`; do # 處理 target/ 目錄下的子目錄。
    echo "Processing directory: target/${dname}"
	mkdir -p "../dataset/target/${dname}"  # 確保目標目錄存在
	echo "Copying .pkl files from ./target/${dname} to ../dataset/target/${dname}/"
	cp "./target/${dname}/"*.pkl "../dataset/target/${dname}/" # 使用 cp 複製檔案，而不是 mv
done