#!/usr/bin/env python3
"""
多模态数据集准备脚本

该脚本用于下载和准备多模态数据集，支持视频-文本-图像三种模态。
使用 MSR-VTT 数据集作为示例，自动下载视频文件、提取关键帧并生成数据清单。

（已针对本地已有数据集的情况优化）

使用方法:
    python prepare_dataset.py --data_dir ./data --keyframe_dir ./keyframes
"""

import os
import json
import cv2
import argparse
import requests
from tqdm import tqdm
from urllib.parse import urlparse
import hashlib
from typing import Dict, List, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MSRVTTDatasetDownloader:
    """MSR-VTT 数据集下载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.video_dir = os.path.join(data_dir, "videos")
        self.annotation_file = os.path.join(data_dir, "train_val_videodatainfo.json")
        
        # 创建目录
        os.makedirs(self.video_dir, exist_ok=True)
        
        # MSR-VTT 数据集信息
        self.dataset_info = {
            "annotation_url": "https://raw.githubusercontent.com/ms-multimedia-challenge/2017-msr-vtt-contest/master/data/train_val_videodatainfo.json",
            "video_base_url": "https://www.robots.ox.ac.uk/~maxbain/frozen-thoughts/msrvtt/videos/",
            "video_extension": ".mp4"
        }
    
    def download_annotation(self) -> bool:
        """下载注释文件（如果本地不存在）"""
        
        # --- 新增修改 ---
        # 检查文件是否已存在
        if os.path.exists(self.annotation_file):
            logger.info(f"找到已存在的注释文件: {self.annotation_file}")
            return True
        # --- 修改结束 ---
            
        try:
            logger.info(f"本地未找到注释文件，正在从 {self.dataset_info['annotation_url']} 下载...")
            response = requests.get(self.dataset_info["annotation_url"], stream=True)
            response.raise_for_status()
            
            with open(self.annotation_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"注释文件已保存到: {self.annotation_file}")
            return True
        except Exception as e:
            logger.error(f"下载注释文件失败: {e}")
            return False
    
    def download_video(self, video_id: str) -> Optional[str]:
        """下载单个视频文件（如果本地不存在）"""
        video_url = f"{self.dataset_info['video_base_url']}{video_id}{self.dataset_info['video_extension']}"
        video_path = os.path.join(self.video_dir, f"{video_id}{self.dataset_info['video_extension']}")
        
        # 如果文件已存在，跳过下载 (此功能原版已有，无需修改)
        if os.path.exists(video_path):
            return video_path
        
        try:
            logger.info(f"本地未找到视频 {video_id}，正在下载...")
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            with open(video_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"下载 {video_id}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"视频已下载: {video_path}")
            return video_path
        except Exception as e:
            logger.error(f"下载视频 {video_id} 失败: {e}")
            return None
    
    def load_annotations(self) -> Dict[str, Any]:
        """加载注释文件"""
        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            logger.info(f"成功加载注释文件，包含 {len(annotations.get('sentences', []))} 个句子")
            return annotations
        except Exception as e:
            logger.error(f"加载注释文件失败: {e}")
            return {}


def extract_keyframe(video_path: str, frame_save_path: str, frame_num: int = 1) -> bool:
    """
    从视频中提取关键帧
    
    Args:
        video_path (str): 视频文件路径
        frame_save_path (str): 关键帧保存路径
        frame_num (int): 要提取的帧数，默认为1（中间帧）
    
    Returns:
        bool: 提取是否成功
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return False
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.error(f"视频文件为空: {video_path}")
            cap.release()
            return False
        
        # 计算要提取的帧位置
        if frame_num == 1:
            # 提取中间帧
            frame_indices = [total_frames // 2]
        else:
            # 均匀分布提取多帧
            frame_indices = [int(i * total_frames / frame_num) for i in range(frame_num)]
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(frame_save_path), exist_ok=True)
        
        extracted_frames = []
        for i, frame_idx in enumerate(frame_indices):
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 生成保存路径
                if frame_num == 1:
                    save_path = frame_save_path
                else:
                    name, ext = os.path.splitext(frame_save_path)
                    save_path = f"{name}_frame_{i:02d}{ext}"
                
                # 保存帧
                success = cv2.imwrite(save_path, frame)
                if success:
                    extracted_frames.append(save_path)
                    logger.debug(f"关键帧已保存: {save_path}")
                else:
                    logger.error(f"保存关键帧失败: {save_path}")
            else:
                logger.error(f"读取第 {frame_idx} 帧失败")
        
        cap.release()
        
        if extracted_frames:
            logger.info(f"成功提取 {len(extracted_frames)} 个关键帧")
            return True
        else:
            logger.error("未能提取任何关键帧")
            return False
            
    except Exception as e:
        logger.error(f"提取关键帧时发生错误: {e}")
        return False


def process_dataset(data_dir: str, keyframe_dir: str, max_samples: Optional[int] = None) -> None:
    """
    处理数据集，生成多模态数据清单
    
    Args:
        data_dir (str): 数据集根目录
        keyframe_dir (str): 关键帧保存目录
        max_samples (int, optional): 最大处理样本数，用于测试
    """
    # 创建关键帧目录
    os.makedirs(keyframe_dir, exist_ok=True)
    
    # 初始化下载器
    downloader = MSRVTTDatasetDownloader(data_dir)
    
    # 下载注释文件（如果不存在）
    if not downloader.download_annotation():
        logger.error("无法下载或找到注释文件，退出")
        return
    
    # 加载注释
    annotations = downloader.load_annotations()
    if not annotations:
        logger.error("无法加载注释文件，退出")
        return
    
    # 处理视频和句子
    videos = annotations.get('videos', [])
    sentences = annotations.get('sentences', [])
    
    # 创建视频ID到信息的映射
    video_info = {video['video_id']: video for video in videos}
    
    # 按数据集分割组织句子
    train_samples = []
    val_samples = []
    
    # 统计信息
    stats = {
        'total_sentences': len(sentences),
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'skipped_download': 0
    }
    
    logger.info(f"开始处理 {len(sentences)} 个句子...")
    
    for sentence in tqdm(sentences, desc="处理样本"):
        if max_samples and stats['processed'] >= max_samples:
            break
            
        video_id = sentence['video_id']
        caption = sentence['caption']
        
        # 获取视频信息
        if video_id not in video_info:
            logger.warning(f"未找到视频 {video_id} 的信息")
            stats['skipped'] += 1
            continue
        
        video_data = video_info[video_id]
        
        # 确定数据集分割
        if video_data.get('split') == 'train':
            target_list = train_samples
        elif video_data.get('split') == 'val':
            target_list = val_samples
        else:
            logger.warning(f"未知的数据集分割: {video_data.get('split')}")
            stats['skipped'] += 1
            continue
        
        # 检查本地视频文件是否存在
        video_path_expected = os.path.join(downloader.video_dir, f"{video_id}{downloader.dataset_info['video_extension']}")
        if not os.path.exists(video_path_expected):
            # 尝试下载（如果用户没有提供）
            video_path = downloader.download_video(video_id)
            if not video_path:
                stats['failed'] += 1
                continue
        else:
            # 使用本地已有的视频
            video_path = video_path_expected
            stats['skipped_download'] += 1
        
        
        # 提取关键帧
        keyframe_path = os.path.join(keyframe_dir, f"{video_id}.jpg")
        if not extract_keyframe(video_path, keyframe_path):
            stats['failed'] += 1
            continue
        
        # 构建样本信息
        sample_info = {
            "text": {"text": caption},
            "image": {"file": os.path.relpath(keyframe_path, data_dir)},
            "video": {"file": os.path.relpath(video_path, data_dir)}
        }
        
        target_list.append(sample_info)
        stats['processed'] += 1
        
        # 每处理100个样本输出一次进度
        if stats['processed'] % 100 == 0:
            logger.info(f"已处理 {stats['processed']} 个样本")
    
    # 保存数据清单
    train_manifest_path = os.path.join(data_dir, "train_manifest.json")
    val_manifest_path = os.path.join(data_dir, "val_manifest.json")
    
    with open(train_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(val_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("数据集处理完成!")
    logger.info(f"总句子数: {stats['total_sentences']}")
    logger.info(f"成功处理: {stats['processed']}")
    logger.info(f"处理失败: {stats['failed']}")
    logger.info(f"跳过样本: {stats['skipped']}")
    logger.info(f"跳过下载（本地已存在）: {stats['skipped_download']}")
    logger.info(f"训练样本: {len(train_samples)}")
    logger.info(f"验证样本: {len(val_samples)}")
    logger.info(f"训练清单: {train_manifest_path}")
    logger.info(f"验证清单: {val_manifest_path}")
    logger.info("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态数据集准备脚本")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data",
        help="数据集根目录 (默认: ./data)"
    )
    parser.add_argument(
        "--keyframe_dir", 
        type=str, 
        default="./keyframes",
        help="关键帧保存目录 (默认: ./keyframes)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="最大处理样本数，用于测试 (默认: 处理所有样本)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="启用详细日志输出"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证参数
    if not os.path.exists(args.data_dir):
        logger.info(f"创建数据目录: {args.data_dir}")
        os.makedirs(args.data_dir, exist_ok=True)
    
    # 处理数据集
    try:
        process_dataset(args.data_dir, args.keyframe_dir, args.max_samples)
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()