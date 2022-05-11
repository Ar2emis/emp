# frozen_string_literal: true

class FoursController < ApplicationController
  PLT = Matplotlib::Pyplot
  U = 1.96

  def index; end

  def create
    @dataset_values = Pandas.read_csv(emp_params[:data].path, header: nil, sep: '\s+').values.to_a
    # @dataset_values = @dataset.values.to_a.flatten.map { |s| s.gsub("\t\"", '   ').delete('\"').split(/ {2,}/) }
    nan = @dataset_values.map { |arr| arr.select { |x| x == '?' } }
    indexes = nan.map.with_index { |x, i| i if x == ['?'] }.compact
    indexes.each { |i| @dataset_values.delete_at(i) }
    @column_count = @dataset_values.first.count
    @count = @dataset_values.count
    @correlation_fields = []
    send(params[:part])
  end

  private

  def first
    @dataset_x = @dataset_values.map(&:first).map(&:to_f)
    @dataset_y = @dataset_values.map(&:last).map(&:to_f)
    @correlation_fields << correlation_field('x', 'y')
    @global_statistics = { x: global_statistics(@dataset_x),
                           y: global_statistics(@dataset_y) }
    @pearson = pearson(@dataset_x, @dataset_y)
    @spirmen = spirmen
    @kendall = kendall
    @corelation_relation = corelation_relation
    @corelation_pearson = @corelation_relation[:statistic].abs > @corelation_relation[:quantile] ? corelation_pearson : nil
  end

  def second
    @matrix = []
    @assymetries = []
    @excesses = []
    (0...(@column_count.pred)).each do |i|
      @matrix[i] = []
      (0...@column_count.pred).each do |j|
        @dataset_x = @dataset_values.map { |arr| arr[i].to_f }
        @dataset_y = @dataset_values.map { |arr| arr[j].to_f }
        @correlation_fields << correlation_field(i.next, j.next)
        # @matrix[i][j] = send(%w[spirmen kendall corelation_relation].sample)
        @matrix[i][j] = if i == j then { value: 1.0 }
                        else
                          corelation = corelation_relation
                          corelation[:statistic].abs > corelation[:quantile] ? corelation : { value: 0.0 }
                        end
      end
      @assymetries << assymetry
      @excesses << excess
    end
  end

  def assymetry
    mean = @dataset_x.sum.to_f / @count
    std_moved = Math.sqrt(@dataset_x.sum { |value| (value - mean)**2 }.to_f / @count)
    assymetry = (Math.sqrt(@count * @count.pred) / (@count - 2)) * (@dataset_x.sum { |value| (value - mean)**3 }.to_f / @count / (std_moved**3))
    # assymetry_std = Math.sqrt((6 * @count * @count.pred).to_f / ((@count - 2) * @count.next * (@count + 3)))
    # assymetry / assymetry_std
  end

  def excess
    mean = @dataset_x.sum.to_f / @count
    std_moved = Math.sqrt(@dataset_x.sum { |value| (value - mean)**2 }.to_f / @count)
    excess_moved = (@dataset_x.sum { |value| (value - mean)**4 }.to_f / @count / (std_moved**4)) - 3
    excess = ((@count**2).pred.to_f / (@count - 2) / (@count - 3)) * (excess_moved + (6.0 / @count.next))
    # excess_std = Math.sqrt((24 * @count * (@count.pred**2)).to_f / ((@count - 2) * (@count - 3) * (@count + 3) * (@count + 5)))
    # excess / excess_std
  end

  def correlation_field(x, y) # rubocop:disable Naming/MethodParameterName
    PLT.scatter(@dataset_x, @dataset_y)
    PLT.ylabel(y)
    PLT.xlabel(x)
    plot_image
  end

  # rubocop:disable Layout/LineLength

  def global_statistics(values)
    # count = values.count
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
    ranks_x = ranks(@dataset_x)
    ranks_y = ranks(@dataset_y)
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
    ranks_x = ranks(@dataset_x)
    ranks_y = ranks(@dataset_y)
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
    h = (@dataset_x.max - @dataset_x.min) / k
    x_sort = @dataset_x.sort
    classes = Array.new(k) do |index|
      klass = [x_sort.first + (index * h), x_sort.first + (index.next * h)]
      index == k.pred ? (klass.first..klass.last) : (klass.first...klass.last)
    end
    @classes_x = Array.new(k) { [] }
    @classes_y = Array.new(k) { [] }
    classes.each_with_index do |klass, index|
      @dataset_x.zip(@dataset_y).each do |x, y|
        if klass.cover?(x)
          @classes_x[index] << x
          @classes_y[index] << y
        end
      end
    end
    @classes_x = @classes_x.select(&:any?)
    @classes_y = @classes_y.select(&:any?)
    @new_x = @classes_x.map { |klass| Array.new(klass.count) { |_| klass.sum.to_f / klass.count } }.flatten
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
    pearson = pearson(@new_x, @classes_y.flatten)[:value]
    first_part = ((@corelation_relation[:value]**2) - (pearson**2)) / (@corelation_relation[:k] - 2)
    last_part = (1 - (@corelation_relation[:value]**2)) / (@count - @corelation_relation[:k])
    { pearson: pearson,
      value: first_part / last_part }
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
    ranks = (1..@count).to_a
    sort = values.sort
    values.map do |elem|
      count = sort.count(elem)
      indicies = (0..@count.pred).select { |i| sort[i] == elem }
      indicies.sum { |i| ranks[i] }.to_f / count
    end
  end

  def equal_ranks(ranks)
    ranks.uniq.map { |v| [v, ranks.count(v)] }.select { |_, count| count > 1 } # x, Aj || y, Bk
  end

  def emp_params
    params.permit(:data, :part)
  end

  def plot_image
    filename = Rails.root.join("tmp/#{SecureRandom.hex}")
    PLT.savefig(File.new(filename, 'wb'))
    "data:image/png;base64,#{Base64.strict_encode64(File.read(filename))}"
  ensure
    File.delete(filename)
    PLT.clf
  end

  def classes_gistogram
    PLT.ylabel = 'p'
    PLT.xlabel = 'x'
    m = 1 + (1.44 * Math.log(@count)).round
    PLT.hist(@dataset_x, bins: m, weights: Array.new(@count) { 1.0 / @count })
    plot_image
  end
end
