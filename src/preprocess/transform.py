from torchvision import transforms

transforms_dict = {
    "colorjitter": transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
    "randomgrayscle": transforms.RandomGrayscale(3, p=0.1),
    "grayscale":transforms.Grayscale(3),
    "randomposterize": transforms.RandomPosterize(4, p=0.5),
    "randomsolarize": transforms.RandomSolarize(128, p=0.5),
    "randomadjustsharpness": transforms.RandomAdjustSharpness(0.3, p=0.5),
    "randomautocontrast":transforms.RandomAutocontrast(p=0.5),
    "randomequalize":transforms.RandomEqualize(p=0.5),
}