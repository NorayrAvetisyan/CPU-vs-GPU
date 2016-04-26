
enum ComputationType{CPUComputing, GPUComputing};

class ImageManager{
public:
    static ImageManager* instance();
    ~ImageManager();
    void PrepareForComputation(ComputationType computType);
private:
    ImageManager();
    ImageManager(const ImageManager& obj);
    ImageManager& operator=(const ImageManager& obj);
    void PrepareImage();
};