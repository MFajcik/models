/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include <fstream>
#include <bitset>
#include <stdlib.h>
#include <math.h>
#include <list>

namespace tensorflow {

// Number of examples to precalculate.
const int kPrecalc = 3000;
// Number of words to read into a sentence before processing.
const int kSentenceSize = 1000;

//#define ORIGINAL
//#define VERBOSE
//#define VERBOSE_SCAN
//#define VERBOSE_BINSERT

#ifndef ORIGINAL
    typedef struct string_with_position{
        std::string str;
        uint64 pos;
    }str_with_pos;
#endif

namespace {

#ifndef ORIGINAL
    bool ScanWord(std::ifstream& fin, str_with_pos* word){
        if (fin.good()){
            fin >> word->str;
            word->pos = fin.tellg();
            if (word->pos == -1) return false;
            return true;
        }
        return false;
    }

#else
    bool ScanWord(StringPiece* input, string* word) {
      str_util::RemoveLeadingWhitespace(input);
      StringPiece tmp;
      if (str_util::ConsumeNonWhitespace(input, &tmp)) {
        word->assign(tmp.data(), tmp.size());
        return true;
      } else {
        return false;
      }

    }
#endif

}  // end namespace

  #ifndef ORIGINAL
  std::unordered_map<int32, string> id2w;
  #endif

class SkipgramWord2vecOp : public OpKernel {
 public:
  explicit SkipgramWord2vecOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), rng_(&philox_) {
    string filename;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("subsample", &subsample_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gradient_ranking",&gradient_ranking_));
    OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
    mutex_lock l(mu_);
    example_pos_ = corpus_size_;
    label_pos_ = corpus_size_;
    label_limit_ = corpus_size_;
    sentence_index_ = kSentenceSize;
    #ifdef VERBOSE
        std::cout<<"Precalculating initial examples!\n";
    #endif
    for (int i = 0; i < kPrecalc; ++i) {
      #ifdef VERBOSE
        std::cout<<"Example n_!"<<i<<"\n";
      #endif
      NextExample(&precalc_examples_[i].input, &precalc_examples_[i].label,
                  &precalc_examples_[i].input_pos, &precalc_examples_[i].label_pos);
    }
    #ifdef VERBOSE
        std::cout<<"Initial precalculate finished!\n";
    #endif
//    std::cout<<"Dumping from 'SkipgramWord2vecOp::constructor' kernel\n";
//    for (int64 i=0; i<precalc_examples_.size(); i++ ){
//        int32 example = precalc_examples_[i].input;
//        uint64 example_pos = precalc_examples_[i].input_pos;
//        int32 label = precalc_examples_[i].label;
//        uint64 label_pos = precalc_examples_[i].label_pos;
//        std::cout<<"\nIE: "<<example<<" E: "<<id2w[example]<<" P: "<<example_pos;
//        std::cout<<"\nIL: "<<label<<" E: "<<id2w[label]<<" P: "<<label_pos<<"\n";
//    }
//    exit(0);
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();

    Tensor example_positions(DT_UINT64, TensorShape({batch_size_}));
    auto Texample_positions = example_positions.flat<uint64>();

    Tensor label_positions(DT_UINT64, TensorShape({batch_size_}));
    auto Tlabel_positions = label_positions.flat<uint64>();

    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; ++i) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        Texample_positions(i) = precalc_examples_[precalc_index_].input_pos;
        Tlabel_positions(i) = precalc_examples_[precalc_index_].label_pos;
        precalc_index_++;
        if (precalc_index_ >= kPrecalc) {
          #ifdef VERBOSE
            std::cout<<"Precalculating new examples!\n";
          #endif
          precalc_index_ = 0;
          for (int j = 0; j < kPrecalc; ++j) {
            #ifdef VERBOSE
                std::cout<<"Example n_!"<<j<<"\n";
            #endif
            NextExample(&precalc_examples_[j].input, &precalc_examples_[j].label,
                  &precalc_examples_[j].input_pos, &precalc_examples_[j].label_pos);
          }
          #ifdef VERBOSE
            std::cout<<"Precalculate finished!\n";
          #endif
        }
      }
      words_per_epoch.scalar<int64>()() = corpus_size_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
    }

    ctx->set_output(0, word_);
    ctx->set_output(1, freq_);
    ctx->set_output(2, words_per_epoch);
    ctx->set_output(3, current_epoch);
    ctx->set_output(4, total_words_processed);
    ctx->set_output(5, examples);
    ctx->set_output(6, labels);
    ctx->set_output(7, example_positions);
    ctx->set_output(8, label_positions);
  }

 private:
  struct Example {
    int32 input;
    int32 label;
    uint64 input_pos;
    uint64 label_pos;
  };

  int32 batch_size_ = 0;
  int32 window_size_ = 5;
  float subsample_ = 1e-3;
  bool gradient_ranking_ = false;
  int min_count_ = 5;
  int32 vocab_size_ = 0;
  Tensor word_;
  Tensor freq_;
  int64 corpus_size_ = 0;
  std::vector<int32> corpus_;
  std::vector<uint64> corpus_pos;

  std::vector<Example> precalc_examples_;
  int precalc_index_ = 0;
  std::vector<int32> sentence_;
  std::vector<uint64> sentence_pos;
  int sentence_index_ = 0;

  mutex mu_;
  random::PhiloxRandom philox_ GUARDED_BY(mu_);
  random::SimplePhilox rng_ GUARDED_BY(mu_);
  int32 current_epoch_ GUARDED_BY(mu_) = -1;
  int64 total_words_processed_ GUARDED_BY(mu_) = 0;
  int64 example_pos_ GUARDED_BY(mu_);
  int32 label_pos_ GUARDED_BY(mu_);
  int32 label_limit_ GUARDED_BY(mu_);

  // {example_pos_, label_pos_} is the cursor for the next example.
  // example_pos_ wraps around at the end of corpus_. For each
  // example, we randomly generate [label_pos_, label_limit) for
  // labels.
  void NextExample(int32* example, int32* label,
        uint64* example_pos, uint64* label_pos ) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    while (true) {
      if (label_pos_ >= label_limit_) {
        ++total_words_processed_;
        ++sentence_index_;
        if (sentence_index_ >= kSentenceSize) {
          sentence_index_ = 0;
          for (int i = 0; i < kSentenceSize; ++i, ++example_pos_) {
            if (example_pos_ >= corpus_size_) {
              ++current_epoch_;
              example_pos_ = 0;
            }
            if (subsample_ > 0) {
              int32 word_freq = freq_.flat<int32>()(corpus_[example_pos_]);
              // See Eq. 5 in http://arxiv.org/abs/1310.4546
              float keep_prob =
                  (std::sqrt(word_freq / (subsample_ * corpus_size_)) + 1) *
                  (subsample_ * corpus_size_) / word_freq;
              if (rng_.RandFloat() > keep_prob) {
                i--;
                continue;
              }
            }
            //std::cout<<id2w[corpus_[example_pos_]]<<"\n";
            sentence_[i] = corpus_[example_pos_];
            sentence_pos[i] = corpus_pos[example_pos_];
          }
        }
        const int32 skip = 1 + rng_.Uniform(window_size_);
        label_pos_ = std::max<int32>(0, sentence_index_ - skip);
        label_limit_ =
            std::min<int32>(kSentenceSize, sentence_index_ + skip + 1);
      }
      if (sentence_index_ != label_pos_) {
        break;
      }
      ++label_pos_;
    }

    *example = sentence_[sentence_index_];
    *label = sentence_[label_pos_++];
    *example_pos = sentence_pos[sentence_index_];
    *label_pos = sentence_pos[label_pos_-1];
    #ifdef VERBOSE
        std::cout<<"\nIE: "<<*example  <<"E:"<<id2w[*example]<<" pos:"<<sentence_pos[sentence_index_]
        <<"\nIL: "<<*label  <<"L:"<<id2w[*label]<<" pos:"<<sentence_pos[label_pos_-1]<<"\n\n";
    #endif
  }

  Status Init(Env* env, const string& filename) {

    #ifndef ORIGINAL
        std::ifstream fin (filename,std::ifstream::in | std::ifstream::binary);
    #else
        string data;
        TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
        StringPiece input = data;
    #endif

    corpus_size_ = 0;
    std::unordered_map<string, int32> word_freq;
    #ifndef ORIGINAL
        str_with_pos w;
        while (ScanWord(fin, &w)) {
          ++(word_freq[w.str]);
          ++corpus_size_;
        }
    #else
        string w;
        while (ScanWord(&input, &w)) {
          ++(word_freq[w]);
          ++corpus_size_;
        }
    #endif

    if (corpus_size_ < window_size_ * 10) {
      return errors::InvalidArgument("The text file ", filename,
                                     " contains too little data: ",
                                     corpus_size_, " words");
    }
    typedef std::pair<string, int32> WordFreq;
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p);
    }

    #ifndef ORIGINAL
        tensorflow::uint64 file_size;
        env->GetFileSize(filename, &file_size);
        std::cout << "Data file: " << filename << " contains " << file_size
                  << " bytes,\n" << corpus_size_ << " words,\n" << word_freq.size()
                  << " unique words,\n" << ordered.size()
                  << " unique frequent words.\n";
    #else
        tensorflow::uint64 file_size;
        env->GetFileSize(filename, &file_size);
        LOG(INFO) << "Data file: " << filename << " contains " << data.size()
                  << " bytes, " << corpus_size_ << " words, " << word_freq.size()
                  << " unique words, " << ordered.size()
                  << " unique frequent words.";
    #endif
    word_freq.clear();
    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              });
    vocab_size_ = static_cast<int32>(1 + ordered.size());
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<string>()(0) = "UNK";
    static const int32 kUnkId = 0;
    std::unordered_map<string, int32> word_id;
    int64 total_counted = 0;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
      const auto& w = ordered[i].first;
      auto id = i + 1;
      word.flat<string>()(id) = w;
      auto word_count = ordered[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
      #ifndef ORIGINAL
      id2w[id]=w;
      #endif
    }
    freq.flat<int32>()(kUnkId) = corpus_size_ - total_counted;
    word_ = word;
    freq_ = freq;
    corpus_.reserve(corpus_size_);
    corpus_pos.reserve(corpus_size_);

    #ifndef ORIGINAL
        fin.clear();// clear fail and eof bits
        fin.seekg(0, std::ios::beg);
        while (ScanWord(fin, &w)) {
            corpus_.push_back(gtl::FindWithDefault(word_id, w.str, kUnkId));
            #ifdef VERBOSE_SCAN
                std::cout<<"Scanned word: "<<w.str<<" pos: "<<w.pos<<"\n";
            #endif
            corpus_pos.push_back(w.pos);
        }
    #else
        input=data;
        while (ScanWord(&input, &w)) {
            corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
        }
    #endif

    precalc_examples_.resize(kPrecalc);
    sentence_.resize(kSentenceSize);
    sentence_pos.resize(kSentenceSize);
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("SkipgramWord2vec").Device(DEVICE_CPU), SkipgramWord2vecOp);
const int GRADIENT_MATRIX_SIZE = 1024;
class NegTrainWord2vecOp : public OpKernel {
 public:

#ifndef ORIIGNAL
  typedef struct Update_s{
    float gradient;
    int32 example;
    uint64 example_pos;
    Update_s(float g,int32 e,uint64 ep) : gradient(g), example(e), example_pos(ep){}
  }Update;
  std::list<Update> batch_updates;
#endif


  explicit NegTrainWord2vecOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    base_.Init(0, 0);

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_negative_samples", &num_samples_));

    std::vector<int32> vocab_count;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_count", &vocab_count));


    std::vector<float> vocab_weights;
    vocab_weights.reserve(vocab_count.size());
    //Unigram distribution ^ 3/4
    for (const auto& f : vocab_count) {
      float r = std::pow(static_cast<float>(f), 0.75f);
      vocab_weights.push_back(r);
    }
    sampler_ = new random::DistributionSampler(vocab_weights);
  }

  ~NegTrainWord2vecOp() { delete sampler_; }

   void insert_into_ordered_tensor(
     float embedding_gradient,
     int example,
     int example_pos,
     Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>& t_grad_ladder,
     Eigen::TensorMap<Eigen::Tensor<int, 2, 1, long int>, 16, Eigen::MakePointer>& t_pos_ladder){
        //binary search and insert
        int top = GRADIENT_MATRIX_SIZE;
        int bottom = 0;
        while (true){
            int split = (top +bottom) / 2;
            #ifdef VERBOSE_BINSERT
                std::cout<<"Split: "<<split<<" TOP: "<<top<<" BOTTOM: "<<bottom<<"\n";
            #endif
            float mid_val = t_grad_ladder(example,split);
            float left_to_mid_val = split>0 ? t_grad_ladder(example,split-1):0.;
            if (split==0 ||
            (left_to_mid_val>embedding_gradient && mid_val<embedding_gradient)) {
                int from = split>0?split:1;
                //shift tensor row
                //TODO: THIS MUST BE DONE IN BETTER WAY
//                for (int i = GRADIENT_MATRIX_SIZE-1; i>=from; i--){
//                    t_grad_ladder(example,i) = t_grad_ladder(example,i-1);
//                    t_pos_ladder(example,i) = t_pos_ladder(example,i-1);
//                }
//                t_grad_ladder(example,split) = embedding_gradient;
//                t_pos_ladder(example,split) = example_pos;

                break;
            }
            else if (split == GRADIENT_MATRIX_SIZE-1 && mid_val> embedding_gradient) break;
            else if (mid_val<=embedding_gradient) top = split;
            else if (mid_val>embedding_gradient) bottom = split;
        }
   }

    void Compute(OpKernelContext* ctx) override {
        static int last_example = -1;
        static int last_example_pos = -1;
        static float example_samples = .0f;
        static float embedding_gradient = 0.0;
        Tensor w_in = ctx->mutable_input(0, false);
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(w_in.shape()),
                    errors::InvalidArgument("Must be a matrix"));
        Tensor w_out = ctx->mutable_input(1, false);
        OP_REQUIRES(ctx, w_in.shape() == w_out.shape(),
                    errors::InvalidArgument("w_in.shape == w_out.shape"));
        const Tensor& examples = ctx->input(2);
        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(examples.shape()),
                    errors::InvalidArgument("Must be a vector"));
        const Tensor& labels = ctx->input(3);
        OP_REQUIRES(ctx, examples.shape() == labels.shape(),
                    errors::InvalidArgument("examples.shape == labels.shape"));

        const Tensor& example_positions = ctx->input(4);
        const Tensor& label_positions = ctx->input(5);
        OP_REQUIRES(ctx, example_positions.shape() == label_positions.shape(),
                    errors::InvalidArgument("example_positions.shape == label_positions.shape"));

        const Tensor& learning_rate = ctx->input(6);
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(learning_rate.shape()),
                    errors::InvalidArgument("Must be a scalar"));

        tensorflow::Tensor pos_ladder = ctx->mutable_input(7, false);
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(pos_ladder.shape()),
                    errors::InvalidArgument("Must be a matrix"));

        tensorflow::Tensor gradient_ladder = ctx->mutable_input(8, false);
        OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(pos_ladder.shape()),
                    errors::InvalidArgument("Must be a matrix"));

        auto Tw_in = w_in.matrix<float>();
        auto Tw_out = w_out.matrix<float>();
        auto Texamples = examples.flat<int32>();
        auto lr = learning_rate.scalar<float>()();
        auto Tlabels = labels.flat<int32>();
        auto Tpos_ladder = pos_ladder.matrix<int32>();
        auto Tgradient_ladder = gradient_ladder.matrix<float>();
        auto Texample_positions = example_positions.flat<uint64>();
        auto Tlabel_positions = label_positions.flat<uint64>();

        //The infered type of examples should be:
        //Eigen::TensorMap<Eigen::Tensor<const int, 1, 1, long int>, 16, Eigen::MakePointer>
        //dump examples and positions
        //    std::cout<<"Dumping from negasmpl\n";
        //    for (int64 i=0; i<Texamples.size(); i++ ){
        //        int32 example = Texamples(i);
        //        uint64 example_pos = Texample_positions(i);
        //        int32 label = Tlabels(i);
        //        uint64 label_pos = Tlabel_positions(i);
        //        std::cout<<"\nIE: "<<example<<" E: "<<id2w[example]<<" P: "<<example_pos;
        //        std::cout<<"\nIL: "<<label<<" E: "<<id2w[label]<<" P: "<<label_pos<<"\n";
        //    }
        //    exit(0);

        const int64 vocab_size = w_in.dim_size(0);
        const int64 dims = w_in.dim_size(1);
        const int64 batch_size = examples.dim_size(0);
        OP_REQUIRES(ctx, vocab_size == sampler_->num(),
                    errors::InvalidArgument("vocab_size mismatches: ", vocab_size,
                                            " vs. ", sampler_->num()));

        // Gradient accumulator for v_in.
        Tensor buf(DT_FLOAT, TensorShape({dims}));
        auto Tbuf = buf.flat<float>();

        // Scalar buffer to hold sigmoid(+/- dot).
        Tensor g_buf(DT_FLOAT, TensorShape({}));
        auto g = g_buf.scalar<float>();

        // The following loop needs 2 random 32-bit values per negative
        // sample.  We reserve 8 values per sample just in case the
        // underlying implementation changes.
        auto rnd = base_.ReserveSamples32(batch_size * num_samples_ * 8);
        random::SimplePhilox srnd(&rnd);


        int ex_cnt = 0;
        for (int64 i = 0; i < batch_size; ++i) {
          const int32 example = Texamples(i);
          DCHECK(0 <= example && example < vocab_size) << example;
          const int32 label = Tlabels(i);
          DCHECK(0 <= label && label < vocab_size) << label;
          const uint64 label_pos = Tlabel_positions(i);
          const uint64 example_pos = Texample_positions(i);


          //get the row from the embedding matrix
          auto v_in = Tw_in.chip<0>(example);
          //Eigen::TensorChippingOp<0l, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer> >
          //ladder = Tpos_ladder.chip<0>(example);

          // Positive: example predicts label.
          //   forward: x = v_in' * v_out
          //            l = log(sigmoid(x))
          //   backward: dl/dx = g = sigmoid(-x)
          //             dl/d(v_in) = g * v_out'
          //             dl/d(v_out) = v_in' * g
          {
            //get the row from similarity matrix
            auto v_out = Tw_out.chip<0>(label);
            //dot product of two rows
            auto dot = (v_in * v_out).sum();
            //sigmoid 1/(1+exp(uv))
            g = (dot.exp() + 1.f).inverse();
            Tbuf = v_out * (g() * lr);
            //update positive similarity row in similarity matrix right away
            v_out += v_in * (g() * lr);
          }

          // Negative samples:
          //   forward: x = v_in' * v_sample
          //            l = log(sigmoid(-x))
          //   backward: dl/dx = g = -sigmoid(x)
          //             dl/d(v_in) = g * v_out'
          //             dl/d(v_out) = v_in' * g
          for (int j = 0; j < num_samples_; ++j) {
            const int sample = sampler_->Sample(&srnd);
            if (sample == label) continue;  // Skip.
            auto v_sample = Tw_out.chip<0>(sample);
            auto dot = (v_in * v_sample).sum();
            g = -((-dot).exp() + 1.f).inverse();
            Tbuf += v_sample * (g() * lr);
            //update negative similarity rows in similarity matrix right away
            v_sample += v_in * (g() * lr);
          }

          // Applies the gradient on v_in.
          v_in += Tbuf;


          ///Extension by Martin Fajcik
          ///Create gradient ladder for each word
          ///Create position ladder according to gradient ladder for each word

          //Inferred types summary:

          //v_in has type
          //class Eigen::TensorChippingOp<0l, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer> >

          //Tbuf has type
          //class Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>

          //Tgradient_ladder has type
          //Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long int>, 16, Eigen::MakePointer>

          //Tpos_ladder has tyoe
          //Eigen::TensorMap<Eigen::Tensor<int, 2, 1, long int>, 16, Eigen::MakePointer>
          #ifndef ORIGINAL
              if (last_example >=0 && (example!= last_example)){
                  embedding_gradient = fabs(embedding_gradient)/example_samples;
                  Update batch_update(embedding_gradient, last_example, last_example_pos);
                  batch_updates.push_back(batch_update);
                  ex_cnt++;
                  //insert_into_ordered_tensor(embedding_gradient, last_example, last_example_pos, Tgradient_ladder,Tpos_ladder);

                  #ifdef VERBOSE
                      std::cout<<"UPDATED: "<<last_example<<" E: "<<id2w[last_example]<<" P: "<<last_example_pos<<" G: "<<embedding_gradient<<"\n";

                      for (int i = 0; i<GRADIENT_MATRIX_SIZE; i++)
                           std::cout<<Tgradient_ladder(last_example,i)<<", ";

                      std::cout<<"\n\n";
                      for (int i = 0; i<GRADIENT_MATRIX_SIZE; i++)
                           std::cout<<Tpos_ladder(last_example,i)<<", ";
                      std::cout<<"\n\n";
                  #endif
                  example_samples = 0.f;
                  embedding_gradient=0.f;
              }
              float gradient_from_sample = ((Eigen::Tensor<float, 0, 1, long int> )Tbuf.sum())(0);
              #ifdef VERBOSE
                std::cout<<"Cumulating gradient "<<gradient_from_sample<<" for "<< id2w[example] <<"\n";
              #endif

              embedding_gradient+= gradient_from_sample;
              example_samples+=1.f;
              last_example = example;
              last_example_pos = example_pos;
           #endif
        }//batch

        Tensor* output_tensor_examples = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ex_cnt}),
                                                         &output_tensor_examples));
        auto o_examples = output_tensor_examples->flat<int32>();


        Tensor* output_tensor_example_positions = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({ex_cnt}),
                                                         &output_tensor_example_positions));
        auto o_example_positions = output_tensor_example_positions->flat<uint64>();


        Tensor* output_tensor_gradients = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({ex_cnt}),
                                                         &output_tensor_gradients));
        auto o_example_gradients = output_tensor_gradients->flat<float>();

        int i = 0;
        for (Update const& update : batch_updates){
            o_examples(i)=update.example;
            o_example_positions(i)=update.example_pos;
            o_example_gradients(i)=update.gradient;
            i++;
        }

        batch_updates.clear();
    }
 private:
  int32 num_samples_ = 0;
  random::DistributionSampler* sampler_ = nullptr;
  GuardedPhiloxRandom base_;
};

REGISTER_KERNEL_BUILDER(Name("NegTrainWord2vec").Device(DEVICE_CPU), NegTrainWord2vecOp);

}  // end namespace tensorflow
