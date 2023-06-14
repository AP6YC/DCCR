        # j.samples = mb
        # y_hats = j.eval("classify(art, samples)")
        # accuracy_score(y_test, y_hat)

# print_allocated_memory()




        # self.preprocessed = preprocessed
        # if not self.preprocessed:
            # rn = resnet50()
            # self.weights = ResNet18_Weights.DEFAULT
            # # rn = resnet18(pretrained=True)
            # rn = resnet18(weights=self.weights)
            # self.mod = create_feature_extractor(rn, {'layer4': 'layer4'})
            # self.mod = self.mod.to('cuda')
            # self.mod.eval()
            # # self.weights = ResNet50_Weights.DEFAULT
            # self.preprocess = self.weights.transforms()
            # self.min = 0.0
            # self.max = 32.0
            # self.mult_factor = 1 / (self.max - self.min) * 2

    # def ext_features(self, img):
    #     with torch.no_grad():
    #         img = img.to('cuda')
    #         prep = self.preprocess(img)
    #         features = self.mod(prep)['layer4']
    #         # avg_features = features.mean(dim=1).flatten(start_dim=1).detach().cpu().numpy()
    #         avg_features = features.detach().mean(dim=1).flatten(start_dim=1)
    #         # avg_features = (avg_features - self.min) / (self.max - self.min) * 2 - 1
    #         avg_features = ((avg_features - self.min) * self.mult_factor - 1) * 3
    #         avg_features = avg_features.sigmoid().cpu().numpy().transpose()
    #         # avg_features = features.mean(dim=1).flatten(start_dim=1).cpu().numpy()
    #         # ipdb.set_trace()
    #         # avg_features.flatten().detach().numpy()

    #     return avg_features