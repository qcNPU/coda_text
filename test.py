import timm

if __name__ == '__main__':
    model_names1 = timm.list_models(pretrained=True)

    strs = '*transformer*t*'
    model_names2 = timm.list_models(strs)

    model_names3 = timm.list_models(strs, pretrained=True)
    print('finish')