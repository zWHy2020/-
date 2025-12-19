"""
多模态数据加载器

支持文本、图像、视频数据的统一加载和预处理。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import json
import os
from torchvision import transforms


class MultimodalDataset(Dataset):
    """
    多模态数据集
    
    支持文本、图像、视频数据的统一处理。
    """
    
    def __init__(
        self,
        data_dir: str,
        data_list: List[Dict[str, Any]],
        text_tokenizer: Optional[Any] = None,
        image_transform: Optional[transforms.Compose] = None,
        video_transform: Optional[transforms.Compose] = None,
        max_text_length: int = 512,
        max_video_frames: int = 10,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.data_dir = data_dir
        self.data_list = data_list
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.video_transform = video_transform
        self.max_text_length = max_text_length
        self.max_video_frames = max_video_frames
        self.image_size = image_size
        
        # 默认图像变换
        if self.image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        
        # 默认视频变换
        if self.video_transform is None:
            self.video_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                
            ])
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict[str, Any]: 样本数据
        """
        item = self.data_list[idx]
        sample = {}
        
        # 文本数据
        if 'text' in item:
            text_data = self._load_text(item['text'])
            if text_data is not None:
                sample['text'] = text_data
        
        # 图像数据
        if 'image' in item:
            image_data = self._load_image(item['image'])
            if image_data is not None:
                sample['image'] = image_data
        
        # 视频数据
        if 'video' in item:
            video_data = self._load_video(item['video'])
            if video_data is not None:
                sample['video'] = video_data
        
        return sample
    
    def _load_text(self, text_info: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        加载文本数据
        
        Args:
            text_info: 文本信息
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: 文本数据
        """
        try:
            if 'file' in text_info:
                # 从文件加载
                text_path = os.path.join(self.data_dir, text_info['file'])
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                # 直接文本
                text = text_info['text']
            
            # 分词
            if self.text_tokenizer:
                tokens = self.text_tokenizer.encode(text, max_length=self.max_text_length, truncation=True)
                input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
                attention_mask = torch.tensor(tokens['attention_mask'], dtype=torch.long)
            else:
                # 简单的字符级编码
                input_ids = torch.tensor([ord(c) for c in text[:self.max_text_length]], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        except Exception as e:
            print(f"加载文本数据时出错: {e}")
            return None
    
    def _load_image(self, image_info: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        加载图像数据
        
        Args:
            image_info: 图像信息
            
        Returns:
            Optional[torch.Tensor]: 图像张量
        """
        try:
            if 'file' in image_info:
                # 从文件加载
                image_path = os.path.join(self.data_dir, image_info['file'])
                image = Image.open(image_path).convert('RGB')
            else:
                # 从数组加载
                image_array = image_info['array']
                image = Image.fromarray(image_array).convert('RGB')
            
            # 应用变换
            image_tensor = self.image_transform(image)
            return image_tensor
        except Exception as e:
            print(f"加载图像数据时出错: {e}")
            return None
    
    def _load_video(self, video_info: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        加载视频数据
        
        Args:
            video_info: 视频信息
            
        Returns:
            Optional[torch.Tensor]: 视频张量
        """
        try:
            if 'file' in video_info:
                # 从文件加载
                video_path = os.path.join(self.data_dir, video_info['file'])
                frames = self._extract_frames_from_video(video_path)
            else:
                # 从数组加载
                frames = video_info['frames']
            
            # 限制帧数
            if len(frames) > self.max_video_frames:
                indices = np.linspace(0, len(frames) - 1, self.max_video_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # 转换为张量
            video_tensor = torch.stack([self.video_transform(frame) for frame in frames])
            return video_tensor
        except Exception as e:
            print(f"加载视频数据时出错: {e}")
            return None
    
    def _extract_frames_from_video(self, video_path: str) -> List[Image.Image]:
        """
        从视频文件提取帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Image.Image]: 帧列表
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
        
        cap.release()
        return frames


class MultimodalDataLoader:
    """
    多模态数据加载器
    
    统一管理文本、图像、视频数据的加载。
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        text_tokenizer: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_text_length: int = 512,
        max_video_frames: int = 10,
        prefetch_factor: int = 2
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.text_tokenizer = text_tokenizer
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.max_video_frames = max_video_frames
        self.prefetch_factor = prefetch_factor
    
    def create_dataset(
        self,
        data_list: List[Dict[str, Any]],
        image_transform: Optional[transforms.Compose] = None,
        video_transform: Optional[transforms.Compose] = None
    ) -> MultimodalDataset:
        """
        创建数据集
        
        Args:
            data_list: 数据列表
            image_transform: 图像变换
            video_transform: 视频变换
            
        Returns:
            MultimodalDataset: 数据集
        """
        return MultimodalDataset(
            data_dir=self.data_dir,
            data_list=data_list,
            text_tokenizer=self.text_tokenizer,
            image_transform=image_transform,
            video_transform=video_transform,
            max_text_length=self.max_text_length,
            max_video_frames=self.max_video_frames,
            image_size=self.image_size
        )
    
    def create_dataloader(
        self,
        dataset: MultimodalDataset,
        shuffle: Optional[bool] = None
    ) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            shuffle: 是否打乱数据
            
        Returns:
            DataLoader: 数据加载器
        """
        if shuffle is None:
            shuffle = self.shuffle
        actual_prefetch_factor = getattr(self, 'prefetch_factor', 2)
        if self.num_workers == 0:
            actual_prefetch_factor = None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_multimodal_batch,
            pin_memory=True,
            prefetch_factor=actual_prefetch_factor,# 预取因子，加速数据加载
            persistent_workers=True if self.num_workers > 0 else False  # 保持工作进程，避免重复创建
        )


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    多模态批次整理函数 - 支持缺失模态的鲁棒处理
    
    Args:
        batch: 批次数据列表
        
    Returns:
        Dict[str, Any]: 整理后的批次数据
    """
    # 分离输入和目标
    inputs = {}
    targets = {}
    attention_masks = {}
    
    # 收集所有模态的数据
    modalities = set()
    for sample in batch:
        modalities.update(sample.keys())
    
    # 处理每个模态
    for modality in modalities:
        modality_data = []
        valid_samples = []
        index_map = {}
        
        # 收集有效样本
        for i, sample in enumerate(batch):
            if modality in sample and sample[modality] is not None:
                index_map[i] = len(modality_data)
                modality_data.append(sample[modality])
                valid_samples.append(i)
        
        # 如果该模态在所有样本中都缺失，跳过
        if not modality_data:
            continue
        
        # 根据模态类型处理数据
        if modality == 'text':
            # 文本数据
            input_ids = [item['input_ids'] for item in modality_data]
            attention_mask = [item['attention_mask'] for item in modality_data]
            
            # 填充到相同长度
            max_len = max(len(ids) for ids in input_ids)
            padded_input_ids = []
            padded_attention_mask = []
            
            for ids, mask in zip(input_ids, attention_mask):
                pad_len = max_len - len(ids)
                padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
            
            # 为缺失的样本创建零填充
            full_batch_input_ids = []
            full_batch_attention_mask = []
            
            for i in range(len(batch)):
                if i in index_map:
                    idx = index_map[i]
                    full_batch_input_ids.append(padded_input_ids[idx])
                    full_batch_attention_mask.append(padded_attention_mask[idx])
                else:
                    # 创建零填充
                    full_batch_input_ids.append(torch.zeros(max_len, dtype=torch.long))
                    full_batch_attention_mask.append(torch.zeros(max_len, dtype=torch.long))
            
            inputs['text_input'] = torch.stack(full_batch_input_ids)
            inputs['text_attention_mask'] = torch.stack(full_batch_attention_mask)
            targets['text'] = inputs['text_input']  # 自监督学习
            
        elif modality == 'image':
            # 图像数据
            images = torch.stack(modality_data)
            
            # 为缺失的样本创建零填充
            full_batch_images = []
            for i in range(len(batch)):
                if i in index_map:
                    idx = index_map[i]
                    full_batch_images.append(images[idx])
                else:
                    # 创建零填充图像
                    zero_image = torch.zeros_like(images[0])
                    full_batch_images.append(zero_image)
            
            inputs['image_input'] = torch.stack(full_batch_images)
            targets['image'] = inputs['image_input']  # 自监督学习
            
        elif modality == 'video':
            # 视频数据
            videos = torch.stack(modality_data)
            
            # 为缺失的样本创建零填充
            full_batch_videos = []
            for i in range(len(batch)):
                if i in index_map:
                    idx = index_map[i]
                    full_batch_videos.append(videos[idx])
                else:
                    # 创建零填充视频
                    zero_video = torch.zeros_like(videos[0])
                    full_batch_videos.append(zero_video)
            
            inputs['video_input'] = torch.stack(full_batch_videos)
            targets['video'] = inputs['video_input']  # 自监督学习
    
    # 构建最终批次
    batch_data = {
        'inputs': inputs,
        'targets': targets
    }
    # 将文本 attention_mask 直接传到 batch 根上，供损失函数使用
    if 'text_attention_mask' in inputs:
        batch_data['attention_mask'] = inputs['text_attention_mask']
    
    return batch_data


def create_sample_data(
    num_samples: int = 100,
    data_dir: str = './sample_data'
) -> List[Dict[str, Any]]:
    """
    创建示例数据
    
    Args:
        num_samples: 样本数量
        data_dir: 数据目录
        
    Returns:
        List[Dict[str, Any]]: 示例数据列表
    """
    os.makedirs(data_dir, exist_ok=True)
    
    data_list = []
    
    for i in range(num_samples):
        sample = {}
        
        # 文本数据
        sample['text'] = {
            'text': f"这是第{i+1}个样本的文本描述。包含农业感知数据的信息。"
        }
        
        # 图像数据
        image_path = os.path.join(data_dir, f'image_{i}.jpg')
        # 创建随机图像
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(image_path)
        sample['image'] = {'file': f'image_{i}.jpg'}
        
        # 视频数据
        video_path = os.path.join(data_dir, f'video_{i}.mp4')
        # 创建随机视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (224, 224))
        for frame_idx in range(30):  # 1秒视频
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        sample['video'] = {'file': f'video_{i}.mp4'}
        
        data_list.append(sample)
    
    return data_list

