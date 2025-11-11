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
            max_num_images=2,
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
            "Image 1",
            "Image 2",
        ]


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



        return images


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

        real_image_idx = ann['real_image']  # 1-based index

        print(real_image_idx-1)
        answer = self.options[real_image_idx-1]
        answer = answer + '<|im_end|>'

        image_placeholders_list = []
        for i in range(num_images):
            img_token = '<|vision_pad|>' * self.num_tokens_per_image
            placeholder = f"Image {i + 1}: <img>{img_token}</img>"
            image_placeholders_list.append(placeholder)

        image_placeholders = '\n'.join(image_placeholders_list)

        question = f"The caption \"{caption}\" describes one pedestrian image.\nAmong the given images, select the one that best matches this caption.Just answer with Image1 and Image2."

        # 构建完整的对话
        full_prompt = f"<|im_start|>user\n{image_placeholders}{question}<|im_end|>\n<|im_start|>assistant\n"

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

        full_input_ids = torch.cat([input_ids, answer_ids], dim=0)

        labels = full_input_ids.clone()
        labels[:len(input_ids)] = -100

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
            "image 1",
            "image 2",
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

        # 2. 构建prompt
        caption = ann['caption']


        real_image_idx = ann['real_image']
        answer = self.options[real_image_idx-1]




        question = f"<image>\n <image>\nThe caption \"{caption}\" describes one pedestrian image.\nAmong the given images, select the one that best matches this caption.Just answer with image1 and image2."


        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image_paths[0]},
                    {"type": "image", "image": image_paths[1]}

                ]
            }
        ]

        return {
            'message': message,
            'answers': answer,
        }


class DocVQADataset(Dataset):
    def __init__(self, split, image_processor, tokenizer, root_path ='../data/CUHK-PEDES/imgs', image_res=224):
        # super().__init__(split)
        self.name = "DocVQA"
        self.image_res = image_res

        self.tokenizer = tokenizer
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

        with open('CUHK_PEDES_new.json', 'r', encoding='utf-8') as f:
            datas = json.load(f)

        self.root_path = root_path

        self.train_datas = []

        if split=="train":
            s = 'train'
        else:
            s='test'

        for item in datas:

            if item['split'] != s:
                continue
            img_path = item.get("file_path")
            if not img_path:
                print(f"⚠️ Warning: Missing 'file_path' in item with id {item.get('id')}, skipping.")
                continue

            # 处理 true questions → Yes
            for q in item.get("true_question", []):
                if not isinstance(q, str):
                    continue

                self.train_datas.append({
                    "image": img_path,
                    'answers':  "Yes.",
                    'question':q,
                })

            # 处理 false questions → No
            for q in item.get("false_question", []):
                if not isinstance(q, str):
                    continue
                self.train_datas.append({
                    "image": img_path,
                    'answers':  "No.",
                    'question':q,
                })

            for q in item.get("might_question", []):
                if not isinstance(q, str):
                    continue
                self.train_datas.append({
                    "image": img_path,
                    'answers': "Possibly.",
                    'question': q,
                })




    def __len__(self):
        return len(self.train_datas)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


    def __getitem__(self, idx):
        example = self.train_datas[idx]
        question = self.correct_casing_finqa(
            example["question"], True
        )

        img_path = os.path.join(self.root_path, example["image"])


        image = Image.open(img_path)
        # image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)

        first_answer = example["answers"]
        answers = first_answer + '<|im_end|>'

        pixel_values = self.transform(image)

        base = self.patch_size * 2
        # image_tokens.append(image.size[0] * image.size[1] // (base * base))
        num_image_tokens = pixel_values.shape[1] * pixel_values.shape[2] // (base * base)

        image_placeholder = f"<img>{'<|vision_pad|>' * num_image_tokens}</img>"
        prompt = f"<|im_start|>user\n{image_placeholder}{question}<|im_end|>\n<|im_start|>assistant\n"

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]
        image_flags = (input_ids == self.image_token_id).int()

        answer_ids = self.tokenizer(
            answers,
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
        }


class DocVQADataset_test(Dataset):
    def __init__(self, split, image_processor, tokenizer, root_path='../data/CUHK-PEDES/imgs',
                 image_res=224):
        # super().__init__(split)
        self.name = "DocVQA"

        self.tokenizer = tokenizer
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

        with open('CUHK_PEDES_new.json', 'r', encoding='utf-8') as f:
            datas = json.load(f)

        self.root_path = root_path

        self.train_datas = []

        if split == "train":
            s = 'train'
        else:
            s = 'test'

        for item in datas:

            if item['split'] != s:
                continue
            img_path = item.get("file_path")
            if not img_path:
                print(f"⚠️ Warning: Missing 'file_path' in item with id {item.get('id')}, skipping.")
                continue

            # 处理 true questions → Yes
            for q in item.get("true_question", []):
                if not isinstance(q, str):
                    continue

                self.train_datas.append({
                    "image": img_path,
                    'answers': "Yes.",
                    'question': q,
                })

            # 处理 false questions → No
            for q in item.get("false_question", []):
                if not isinstance(q, str):
                    continue
                self.train_datas.append({
                    "image": img_path,
                    'answers':  "No.",
                    'question': q,
                })

            for q in item.get("might_question", []):
                if not isinstance(q, str):
                    continue
                self.train_datas.append({
                    "image": img_path,
                    'answers': "Possibly.",
                    'question': q,
                })



    def __len__(self):
        return len(self.train_datas)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text

    def __getitem__(self, idx):
        example = self.train_datas[idx]
        question = self.correct_casing_finqa(
            example["question"], True
        )

        img_path = os.path.join(self.root_path, example["image"])

        image = Image.open(img_path)

        first_answer = example["answers"]
        # answers = first_answer + '<|im_end|>'

        pixel_values = self.transform(image)

        base = self.patch_size * 2
        # image_tokens.append(image.size[0] * image.size[1] // (base * base))
        num_image_tokens = pixel_values.shape[1] * pixel_values.shape[2] // (base * base)

        # 构建prompt
        image_placeholder = f"<img>{'<|vision_pad|>' * num_image_tokens}</img>"
        prompt = f"<|im_start|>user\n{image_placeholder}{question}<|im_end|>\n<|im_start|>assistant\n"

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids[0]
        image_flags = (input_ids == self.image_token_id).int()


        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'image_flags': image_flags,
            'answer': first_answer
        }

