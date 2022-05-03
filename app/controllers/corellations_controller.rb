# frozen_string_literal: true

class CorellationsController < ApplicationController
  PLT = Matplotlib::Pyplot
  U = 1.96

  def index; end

  def create
    @df = Pandas.read_csv(emp_params[:data].path, header: nil, sep: '\s+')
    @df_values = @df.values.to_a
    @column_count = @df_values.first.count
    @count = @df_values.count
    @correlation_fields = []
    params[:dimensions].blank? ? simple : complex
  end

  private

  def simple
    @df_x = @df_values.map(&:first)
    @df_y = @df_values.map(&:last)
    @correlation_fields << correlation_field_plot('x', 'y')
    @statistics = { x: statistics(@df_x), y: statistics(@df_y) }
    @pearson = pearson(@df_x, @df_y)
    @spirmen = spirmen
    @kendall = kendall
    @corelation_relation = corelation_relation
    @corelation_pearson = @corelation_relation[:statistic].abs > @corelation_relation[:quantile] ? corelation_pearson : nil
  end

  def complex
    @df[3] = Pandas.to_numeric(@df[3], errors: 'coerce')
    @df = @df[@df[3].notnull].drop(8, axis: 1)
    @df_values = @df.values.to_a
    @column_count = @df_values.first.count
    @statistics = {}
    @matrix = Array.new(@column_count) do |index|
      @df_x = @df[index].to_a
      @statistics[index.next] = statistics(@df_x)

      Array.new(@column_count) do |jndex|
        next { value: 1 } if index == jndex

        @df_y = @df[jndex].to_a
        @correlation_fields << correlation_field_plot(index.next, jndex.next)
        @corelation_relation = corelation_relation
        @corelation_relation[:statistic].abs > @corelation_relation[:quantile] ? @corelation_relation : { value: 0 }
      end
    end
  end

  def correlation_field_plot(x, y) # rubocop:disable Naming/MethodParameterName
    PLT.scatter(@df_x, @df_y)
    PLT.ylabel(y)
    PLT.xlabel(x)
    plot_image
  end

  # rubocop:disable Layout/LineLength

  def statistics(values)
    mean = values.sum.to_f / @count
    sorted_values = values.sort
    median = (sorted_values[(sorted_values.size.pred / 2)] + sorted_values[sorted_values.size / 2]).to_f / 2
    std = Math.sqrt(values.sum { |value| (value - mean)**2 }.to_f / @count.pred).to_f
    std_moved = Math.sqrt(values.sum { |value| (value - mean)**2 }.to_f / @count)
    excess_moved = (values.sum { |value| (value - mean)**4 }.to_f / @count / (std_moved**4)) - 3
    mean_std = std / Math.sqrt(@count)
    std_std = std / Math.sqrt(2 * @count)
    assymetry = (Math.sqrt(@count * @count.pred) / (@count - 2)) * (values.sum { |value| (value - mean)**3 }.to_f / @count / (std_moved**3))
    assymetry_std = Math.sqrt((6 * @count * @count.pred).to_f / ((@count - 2) * @count.next * (@count + 3)))
    excess = ((@count**2).pred / (@count - 2) / (@count - 3)) * (excess_moved + (6.0 / @count.next))
    excess_std = Math.sqrt((24 * @count * (@count.pred**2)).to_f / ((@count - 2) * (@count - 3) * (@count + 3) * (@count + 5)))
    {
      mean: { value: mean, std: mean_std, ninety_five: (mean - (2 * mean_std)..mean + mean_std) },
      median: {
        value: median,
        ninety_five: (sorted_values[(@count.to_f / 2) - (1.96 * (Math.sqrt(@count).to_f / 2)).round]..sorted_values[(@count.to_f / 2) + 1 + (1.96 * (Math.sqrt(@count).to_f / 2)).round])
      },
      std: { value: std, std: std_std, ninety_five: (std - (2 * std_std)..std + (2 * std_std)) },
      assymetry: {
        value: assymetry, std: assymetry_std,
        ninety_five: (assymetry - (2 * assymetry_std)..assymetry + (2 * assymetry_std))
      },
      excess: {
        value: excess, std: excess_std,
        ninety_five: (excess - (2 * excess_std)..excess + (2 * excess_std))
      }
    }
  end

  # rubocop:enable Layout/LineLength

  def pearson(values_x, values_y)
    mean_x = values_x.sum.to_f / @count
    mean_y = values_y.sum.to_f / @count
    mean_xy = values_x.zip(values_y).sum { |x, y| x * y }.to_f / @count
    std_moved_x = Math.sqrt(values_x.sum { |value| (value - mean_x)**2 }.to_f / @count)
    std_moved_y = Math.sqrt(values_y.sum { |value| (value - mean_y)**2 }.to_f / @count)
    r = (mean_xy - (mean_x * mean_y)) / (std_moved_x * std_moved_y)
    rn = r + (r * (1 - (r**2)) / (2 * @count)) - (U * (1 - (r**2)) / Math.sqrt(@count.pred))
    rv = r + (r * (1 - (r**2)) / (2 * @count)) + (U * (1 - (r**2)) / Math.sqrt(@count.pred))
    t = (r * Math.sqrt(@count - 2)) / Math.sqrt(1 - (r**2))
    { value: r,
      ninety_five: (rn..rv),
      statistic: t,
      quantile: U }
  end

  def spirmen
    ranks_x = ranks(@df[0])
    ranks_y = ranks(@df[1])
    ranks_different = ranks_x.map.with_index { |rank, index| (rank - ranks_y[index])**2 }.sum.to_f
    nn2_1 = (@count * ((@count**2) - 1)).to_f # rubocop:disable Naming/VariableNumber
    r = if ranks_x.uniq.count == @count && ranks_y.uniq.count == @count # !!!
          1 - ((6.0 / nn2_1) * ranks_different)
        else
          eq_x = equal_ranks(ranks_x)
          eq_y = equal_ranks(ranks_y)
          a = eq_x.sum { |_, aj| (aj**3) - aj }.to_f / 12
          b = eq_y.sum { |_, bk| (bk**3) - bk }.to_f / 12
          ((nn2_1 / 6) - ranks_different - a - b) / Math.sqrt(((nn2_1 / 6) - (2 * a)) * ((nn2_1 / 6) - (2 * b)))
        end
    t = (r * Math.sqrt(@count - 2)) / Math.sqrt(1 - (r**2))
    { value: r,
      statistic: t,
      quantile: U }
  end

  def kendall
    ranks_x = ranks(@df[0])
    ranks_y = ranks(@df[1])
    ranks = ranks_x.zip(ranks_y).sort_by(&:first)
    ranks_x = ranks.map(&:first)
    ranks_y = ranks.map(&:last)
    nn_1_2 = (@count * @count.pred).to_f / 2 # rubocop:disable Naming/VariableNumber
    r = if ranks_x.uniq.count == @count && ranks_y.uniq.count == @count
          s = (0...@count.pred).to_a.sum do |i|
            ((i.next)...@count).to_a.sum do |j|
              if ranks_y[i] < ranks_y[j]
                1
              elsif ranks_y[i] > ranks_y[j]
                -1
              end
            end
          end
          s.to_f / nn_1_2
        else
          s = (0...@count.pred).to_a.sum do |i|
            ((i + 1)...@count).to_a.sum do |j|
              if ranks_y[i] < ranks_y[j] && ranks_x[i] != ranks_x[j]
                1
              elsif ranks_y[i] > ranks_y[j] && ranks_x[i] != ranks_x[j]
                -1
              else
                0
              end
            end
          end
          # binding.pry
          eq_x = equal_ranks(ranks_x)
          eq_y = equal_ranks(ranks_y)
          c = eq_x.sum { |_, aj| aj * aj.pred }.to_f / 2
          d = eq_y.sum { |_, bk| bk * bk.pred }.to_f / 2
          s.to_f / Math.sqrt((nn_1_2 - c) * (nn_1_2 - d))
        end
    t = (3 * r * Math.sqrt(@count * @count.pred)) / Math.sqrt(2 * ((2 * @count) + 5))
    { value: r,
      statistic: t,
      quantile: U }
  end

  def corelation_relation
    k = 1 + (1.44 * Math.log(@count)).round
    h = (@df_x.max - @df_x.min) / k
    x_sort = @df_x.sort
    classes = Array.new(k) do |index|
      klass = [x_sort.first + (index * h), x_sort.first + (index.next * h)]
      index == k.pred ? (klass.first..klass.last) : (klass.first...klass.last)
    end
    @classes_x = Array.new(k) { [] }
    @classes_y = Array.new(k) { [] }
    classes.each_with_index do |klass, index|
      @df_x.zip(@df_y).each do |x, y|
        if klass.cover?(x)
          @classes_x[index] << x
          @classes_y[index] << y
        end
      end
    end
    @classes_y = @classes_y.select(&:any?)
    mean_yl = @classes_y.map { |klass| klass.sum.to_f / klass.count }
    mean_y = @classes_y.sum(&:sum).to_f / @count
    s2_mean_y = @classes_y.map.with_index { |klass, index| klass.count * ((mean_yl[index] - mean_y)**2) }.sum.to_f
    s2_y = @classes_y.sum { |klass| klass.sum { |y| (y - mean_y)**2 } }.to_f
    r = Math.sqrt(s2_mean_y / s2_y)
    t = ((r**2) / k.pred) / ((1 - (r**2)) / (@count - k))
    { value: r,
      statistic: t,
      quantile: fisher_quantile(k, @count - k),
      k: k }
  end

  def corelation_pearson
    first_part = ((@corelation_relation[:value]**2) - (@pearson[:value]**2)) / (@corelation_relation[:k] - 2)
    last_part = (1 - (@corelation_relation[:value]**2)) / (@count - @corelation_relation[:k])
    first_part / last_part
  end

  def fisher_quantile(k, n)
    if n <= 120
      if k <= 5 then 2.29
      elsif k <= 7 then 2.09
      elsif k <= 10 then 1.91
      elsif k <= 11 then 1.87
      elsif k <= 13 then 1.8
      elsif k <= 30 then 1.55
      elsif k <= 60 then 1.43
      elsif k <= 120 then 1.35
      else 1.25
      end
    else
      if k <= 5 then 2.21
      elsif k <= 7 then 2.01
      elsif k <= 10 then 1.83
      elsif k <= 11 then 1.79
      elsif k <= 13 then 1.72
      elsif k <= 30 then 1.56
      elsif k <= 60 then 1.32
      elsif k <= 120 then 1.22
      else 1.0
      end
    end
  end

  def ranks(values)
    values.rank.to_a
  end

  def equal_ranks(ranks)
    ranks.uniq.map { |v| [v, ranks.count(v)] }.select { |_, count| count > 1 }
  end

  def emp_params
    params.permit(:data, :dimensions)
  end

  def plot_image
    filename = Rails.root.join("tmp/#{SecureRandom.hex}")
    PLT.savefig(File.new(filename, 'wb'))
    "data:image/png;base64,#{Base64.strict_encode64(File.read(filename))}"
  ensure
    File.delete(filename)
    PLT.clf
  end
end
