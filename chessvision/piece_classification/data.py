assert False, "This file is not meant to be executed."
import tlc
import torchvision.datasets as datasets
import torchvision.transforms as transforms

TRAIN_DATASET_PATH = r"C:\Data\chessboard-squares\training"
VAL_DATASET_PATH = r"C:\Data\chessboard-squares\validation"

TRAIN_DATASET_NAME = "chessvision-training"
VAL_DATASET_NAME = "chessvision-validation"

train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=0),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ]
)
label_names = [
    "White Bishop",
    "White King",
    "White Knight",
    "White Pawn",
    "White Queen",
    "White Rook",
    "Black Bishop",
    "Black King",
    "Black Knight",
    "Black Pawn",
    "Black Queen",
    "Black Rook",
    "Empty Square",
]


# def loader(path):
#     img = Image.open(path)
#     img.load()  # Don't do .convert(mode) here!
#     return img


train_dataset = datasets.ImageFolder(TRAIN_DATASET_PATH)
val_dataset = datasets.ImageFolder(VAL_DATASET_PATH)

sample_structure = (tlc.PILImage("image"), tlc.CategoricalLabel("label", classes=label_names))


def train_trans(sample):
    return train_transforms(sample[0]), sample[1]


def val_trans(sample):
    return val_transforms(sample[0]), sample[1]


tlc_train_dataset = tlc.Table.from_torch_dataset(
    dataset=train_dataset,
    dataset_name=TRAIN_DATASET_NAME,
    structure=sample_structure,
).map(train_trans)

tlc_val_dataset = tlc.Table.from_torch_dataset(
    dataset=val_dataset,
    dataset_name=VAL_DATASET_NAME,
    structure=sample_structure,
).map(val_trans)

assert True
