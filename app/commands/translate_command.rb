class TranslateCommand
  prepend SimpleCommand

  attr_reader :path

  class << self
    def model
      @model ||= OnnxRuntime::Model.new(Rails.root.join("config/model.onnx"))
    end
  end

  def initialize(path:)
    @path = path
  end

  def call
    doc = HexaPDF::Document.open(path)

      # (0...doc.pages.count).each do |page|
      height, width = image_size

      pipeline = ImageProcessing::Vips.source(path)
      pipeline = pipeline.loader(page: 0)
      pipeline = pipeline.convert("png")

      file = pipeline.call
      image = Vips::Image.new_from_file(file.path)
      original_height, original_width = image.height, image.width

      gain = [ height / original_height.to_d, width / original_width.to_d ].min
      pad_x = ((width - original_width * gain) / 2 - 0.1).round
      pad_y = ((height - original_height * gain) / 2 - 0.1).round

      pipeline = ImageProcessing::Vips.source(file.path)
      pipeline = pipeline.resize_and_pad(height, width)

      file = pipeline.call
      image = Vips::Image.new_from_file(file.path)

      # 将像素值归一化到 0-1 范围
      pixels = image.divide(255.0)

      # 只保留RGB通道
      pixels = pixels.extract_band(0, n: 3)

      # 将图像从 HWC (Height, Width, Channels) 格式转换为 CHW (Channels, Height, Width) 格式
      # 这是因为大多数深度学习框架（如 PyTorch）使用 CHW 格式
      pixels = Numpy.transpose(pixels.to_a, [ 2, 0, 1 ])

      # 添加 batch 维度，变成 BCHW (Batch, Channels, Height, Width) 格式
      pixels = Numpy.expand_dims(pixels, axis: 0)

      predictions = model.predict({ images: pixels }).dig("output0", 0)
      predictions = predictions.map { |it| normalize(prediction: it, gain:, pad_x:, pad_y:) }.compact

      page = doc.pages[0]
      canvas = page.canvas(type: :overlay)


      predictions.each do |it|
        canvas
        .stroke_color("ff0022")
        .line_width(1)
        .rectangle(it[:llx], it[:lly], it[:urx] - it[:llx], it[:ury] - it[:lly])
        .stroke
      end
    # end

    doc.write("tmp/output.pdf")
  end

  private

  def model
    @model ||= self.class.model
  end

  def model_metadata
    @model_metadata ||= model.metadata.with_indifferent_access
  end

  def stride
    @stride ||= model_metadata.dig(:custom_metadata_map, :stride).to_d
  end

  def image_size
    @image_size ||= JSON.parse(model_metadata.dig(:custom_metadata_map, :imgsz))
  end

  def names
    @names ||= JSON.parse(model_metadata.dig(:custom_metadata_map, :names).gsub("'", '"').gsub(/(\d):/, '"\1":'))
  end

  def normalize(prediction:, gain:, pad_x:, pad_y:)
    llx, lly, urx, ury, confidence, type = prediction
    return if confidence < 0.25

    llx = (llx - pad_x) / gain
    lly = (lly - pad_y) / gain
    urx = (urx - pad_x) / gain
    ury = (ury - pad_y) / gain
    name = names[type.to_i.to_s]

    { llx:, lly:, urx:, ury:, name: }
  end
end
