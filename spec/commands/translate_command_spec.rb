require "rails_helper"

describe TranslateCommand do
  subject(:command) { described_class.call(path:) }

  let(:path) { file_fixture("example.pdf") }

  it "succeeds" do
    expect(command).to be_success
  end
end
