def srgan(pretrained=False):
    """
    if pretrained is true then first check of the model
    is already downloaded. If not then download the model
    from cloud and store locally.

    so that if pretrained is true first checks fot the model, 
    if not found then only it downloads


    if pretrained is false then it returns the the untrained model
    of generator and discriminator
    """

    # if pretrained:
    #     if model already in local:
    #         return model
    #     else:
    #         download model 
    #         return model
