import os
import json
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict


class PedestrianMatchingDataset(Dataset):
    """
    行人图文匹配数据集
    给定文本描述和多张候选图像,选择最匹配的图像
    """
    def __init__(
            self,
            data,
            image_processor,
            tokenizer,
            image_res=224,
            max_num_images=5,
            data_root='../../data/CUHK-PEDES/'
    ):
        self.tokenizer = tokenizer
        self.image_res = image_res
        self.max_num_images = max_num_images
        self.data_root = data_root

        # 加载数据
        # with open(data_path, 'r', encoding='utf-8') as f:
        self.data = data

        # 图像预处理
        self.background_color = tuple(int(x * 255) for x in image_processor.image_mean)
        self.transform = T.Compose([
            T.Resize((image_res, image_res), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])

        # Token配置
        self.patch_size = 14
        self.image_token_id = tokenizer.vocab.get('<|vision_pad|>', tokenizer.unk_token_id)

        # 计算每张图像的token数量
        base = self.patch_size * 2
        self.num_tokens_per_image = image_res * image_res // (base * base)
        self.options = [
            "Image A",
            "Image B",
            "Image C",
            "Image D",
            "Image E",
        ]

        self.question_template = '''You are an expert in image-text matching.
        Given a text description and five candidate pedestrian images,
        select the one that best matches the text.

        Text description:
        "{text}"

        Images:
        {image_placeholders}

        Consider clothing, gender, pose, and unique appearance details.
        Output strictly in the format:
        Image {{letter}}
        where {{letter}} is one of A, B, C, D, or E.
        Do not output anything else.
        '''

    def _load_image(self, image_path: str) -> Image.Image:
        """加载单张图像"""

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 创建黑色占位图
            image = Image.new('RGB', (self.image_res, self.image_res),
                              color=self.background_color)

        return image

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """加载多张候选图像"""
        images = []
        for img_path in image_paths[:self.max_num_images]:
            image = self._load_image(img_path)
            images.append(image)

        # 如果图像数量不足,补充空白图像
        while len(images) < self.max_num_images:
            blank = Image.new('RGB', (self.image_res, self.image_res),
                              color=self.background_color)
            images.append(blank)

        return images

    def _create_prompt(self, text: str, num_images: int) -> str:
        """创建包含图像占位符的prompt"""
        # 为每张图像创建占位符
        image_placeholders_list = []
        for i in range(num_images):
            img_token = '<|vision_pad|>' * self.num_tokens_per_image
            placeholder = f"Image {i + 1}: <img>{img_token}</img>"
            image_placeholders_list.append(placeholder)

        image_placeholders = '\n'.join(image_placeholders_list)

        # 填充模板
        prompt = self.question_template.format(
            text=text,
            image_placeholders=image_placeholders
        )

        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        ann = self.data[index]

        # 1. 加载图像
        image_paths = ann['images']
        images = self._load_images(image_paths)
        num_images = min(len(image_paths), self.max_num_images)

        # 处理图像为tensor
        pixel_values_list = []
        for img in images:
            pixel_values = self.transform(img)
            pixel_values_list.append(pixel_values)

        pixel_values = torch.stack(pixel_values_list)  # [max_num_images, C, H, W]

        # 2. 构建prompt
        caption = ann['caption']
        prompt = self._create_prompt(caption, num_images)

        real_image_idx = ann['real_image']  # 1-based index
        answer = self.options[real_image_idx-1]
        answer = answer + '<|im_end|>'

        # 构建完整的对话
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # 3. Tokenize
        input_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        # 4. 创建image_flags
        image_flags = (input_ids == self.image_token_id).int()

        answer_ids = self.tokenizer(
            answer,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        # 拼接input_ids和answer_ids作为完整序列
        full_input_ids = torch.cat([input_ids, answer_ids], dim=0)

        # labels: 只计算answer部分的loss
        labels = full_input_ids.clone()
        labels[:len(input_ids)] = -100  # 忽略prompt部分

        # image_flags也需要扩展
        answer_flags = torch.zeros_like(answer_ids)
        full_image_flags = torch.cat([image_flags, answer_flags], dim=0)

        return {
            'pixel_values': pixel_values,
            'input_ids': full_input_ids,
            'image_flags': full_image_flags,
            'labels': labels,
            'answer': answer,
            'caption': caption,
            'num_images': num_images,
            'real_image_idx': real_image_idx
        }



class PedestrianMatchingDataset_Test(Dataset):
    """
    行人图文匹配数据集
    给定文本描述和多张候选图像,选择最匹配的图像
    """
    def __init__(
            self,
            data,
            image_processor,
            tokenizer,
            image_res=224,
            max_num_images=5,
            data_root='../../data/CUHK-PEDES/'
    ):
        self.tokenizer = tokenizer
        self.image_res = image_res
        self.max_num_images = max_num_images
        self.data_root = data_root

        # 加载数据
        # with open(data_path, 'r', encoding='utf-8') as f:
        self.data = data

        self.options = [
            "Image A",
            "Image B",
            "Image C",
            "Image D",
            "Image E",
        ]

        # 图像预处理
        self.background_color = tuple(int(x * 255) for x in image_processor.image_mean)
        self.transform = T.Compose([
            T.Resize((image_res, image_res), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        ])

        # Token配置
        self.patch_size = 14
        self.image_token_id = tokenizer.vocab.get('<|vision_pad|>', tokenizer.unk_token_id)

        # 计算每张图像的token数量
        base = self.patch_size * 2
        self.num_tokens_per_image = image_res * image_res // (base * base)

        # 问题模板
        self.question_template = '''You are an expert in image-text matching.
        Given a text description and five candidate pedestrian images,
        select the one that best matches the text.

        Text description:
        "{text}"

        Images:
        {image_placeholders}

        Consider clothing, gender, pose, and unique appearance details.
        Output strictly in the format:
        Image {{letter}}
        where {{letter}} is one of A, B, C, D, or E.
        Do not output anything else.
        '''

    def _load_image(self, image_path: str) -> Image.Image:
        """加载单张图像"""
        # 处理相对路径
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 创建黑色占位图
            image = Image.new('RGB', (self.image_res, self.image_res),
                              color=self.background_color)

        return image

    def _load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """加载多张候选图像"""
        images = []
        for img_path in image_paths[:self.max_num_images]:
            image = self._load_image(img_path)
            images.append(image)

        # 如果图像数量不足,补充空白图像
        while len(images) < self.max_num_images:
            blank = Image.new('RGB', (self.image_res, self.image_res),
                              color=self.background_color)
            images.append(blank)

        return images

    def _create_prompt(self, text: str, num_images: int) -> str:
        """创建包含图像占位符的prompt"""
        # 为每张图像创建占位符
        image_placeholders_list = []
        for i in range(num_images):
            img_token = '<|vision_pad|>' * self.num_tokens_per_image
            placeholder = f"Image {i + 1}: <img>{img_token}</img>"
            image_placeholders_list.append(placeholder)

        image_placeholders = '\n'.join(image_placeholders_list)

        # 填充模板
        prompt = self.question_template.format(
            text=text,
            image_placeholders=image_placeholders
        )

        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        ann = self.data[index]

        # 1. 加载图像
        image_paths = ann['images']
        images = self._load_images(image_paths)
        num_images = min(len(image_paths), self.max_num_images)

        # 处理图像为tensor
        pixel_values_list = []
        for img in images:
            pixel_values = self.transform(img)
            pixel_values_list.append(pixel_values)

        pixel_values = torch.stack(pixel_values_list)  # [max_num_images, C, H, W]

        # 2. 构建prompt
        caption = ann['caption']
        prompt = self._create_prompt(caption, num_images)

        real_image_idx = ann['real_image']  # 1-based index
        answer = self.options[real_image_idx-1]


        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # 3. Tokenize
        input_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]

        # 4. 创建image_flags
        image_flags = (input_ids == self.image_token_id).int()



        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'image_flags': image_flags,
            'answer': answer,
            'caption': caption,
            'real_image_idx': real_image_idx
        }