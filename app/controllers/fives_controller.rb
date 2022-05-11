# frozen_string_literal: true

class FivesController < ApplicationController
  PLT = Matplotlib::Pyplot
  T = 2.16

  def index; end

  def create
    @dataset_values = Pandas.read_csv(emp_params[:data].path, header: nil, sep: '\s+').values.to_a
    @column_count = @dataset_values.first.count
    if params[:dependent_indicator].to_i > @column_count || params[:independent_indicator].to_i > @column_count
      raise ArgumentError.new("Out of number of columns (from 1 to #{@column_count})")
    end

    send(params[:part])
  end

  private

  def first
    @count = @dataset_values.count
    @independent_indicator ||= emp_params[:independent_indicator]
    (@dependent_indicator = @independent_indicator.to_i == 1 ? 2 : 1) if emp_params[:dependent_indicator].empty?
    @independent_column = @dataset_values.map { |arr| arr[@dependent_indicator.to_i.pred].to_f }
    @dependent_column = @dataset_values.map { |arr| arr[@independent_indicator.to_i.pred].to_f }
    @mean_x = @dependent_column.sum.to_f / @count
    @mean_y = @independent_column.sum.to_f / @count

    pearson = pearson(@dependent_column, @independent_column)
    @a1 = pearson[:r] * pearson[:std_moved_y] / pearson[:std_moved_x]
    @a0 = @mean_y - (@a1 * @mean_x)

    @se2 = @independent_column.map.with_index { |y, i| (y - (@a0 + (@a1 * @dependent_column[i])))**2 }.sum.to_f / (@count - 2)
    @std_a0 = Math.sqrt(@se2 / @count * (1 + ((@mean_x**2) / (pearson[:std_moved_x]**2))))
    @std_a1 = Math.sqrt(@se2 / (@count * (pearson[:std_moved_x]**2)))
    @a0_interval = { first: @a0 - (T * @std_a0), last: @a0 + (T * @std_a0) }
    @a1_interval = { first: @a1 - (T * @std_a1), last: @a1 + (T * @std_a1) }
    @a0_stat = @a0 / @std_a0
    @a1_stat = @a1 / @std_a1
    @p_a0 = 2 * (1 - ScipyStats.t(@count.pred).cdf(@a0_stat.abs))
    @p_a1 = 2 * (1 - ScipyStats.t(@count.pred).cdf(@a1_stat.abs))
    @determination = (pearson[:r]**2) * 100
    @f_test = @dependent_column.sum { |x| ((@a0 + (@a1 * x)) - @mean_y)**2 }.to_f / @se2
    @p_f_test = 1 - ScipyStats.f(1, @count - 2).cdf(@f_test)
    @e_array = @independent_column.map.with_index { |y, i| y - (@a0 + (@a1 * @dependent_column[i])) }
    @e_assym = assymetry_stat
    @e_excess = excess_stat
    @dfx = @dependent_column.map { |x| (@se2 / @count) + (@std_a1 * ((x - @mean_x)**2)) }
    @fxn = @dfx.map.with_index { |dx, i| @a0 + (@a1 * @dependent_column[i]) - (T * Math.sqrt(dx)) }
    @fxv = @dfx.map.with_index { |dx, i| @a0 + (@a1 * @dependent_column[i]) + (T * Math.sqrt(dx)) }
    @yxn = @dependent_column.map { |x| @a0 + (@a1 * x) - (T * Math.sqrt(@se2)) }
    @yxv = @dependent_column.map { |x| @a0 + (@a1 * x) + (T * Math.sqrt(@se2)) }
    @correlation_field = correlation_field
  end

  def second
    @dependent_indicator = emp_params[:dependent_indicator].empty? ? 1 : params[:dependent_indicator].to_i
    nan = @dataset_values.map { |arr| arr.select { |x| x == '?' } }
    indexes = nan.map.with_index { |x, i| i if x == ['?'] }.compact
    indexes.each { |i| @dataset_values.delete_at(i) }
    @count = @dataset_values.count
    @dataset_values.map! { |arr| arr.map(&:to_f) }
    @matrix = []
    (0...(@column_count.pred)).each do |i|
      @matrix[i] = []
      (0...@column_count.pred).each do |j|
        @dataset_x = @dataset_values.map { |arr| arr[i].to_f }
        @dataset_y = @dataset_values.map { |arr| arr[j].to_f }
        @matrix[i][j] = i == j ? 0.0 : spirmen.abs
      end
    end
    @independent_indicator = @matrix[@dependent_indicator.pred].index(@matrix[@dependent_indicator.pred].max).next
    first
    @y = @dataset_values.map { |arr| arr[@dependent_indicator.pred].to_f }
    # x1_column = Array.new(@count) { 1.0 }
    x_other_matrix = @dataset_values.each { |arr| arr.delete_at(@dependent_indicator.pred) && arr.pop }
    x_other_matrix = StatModels.add_constant(x_other_matrix)
    # @x_matrix = x1_column.zip(x_other_matrix).map(&:flatten)
    # @x_transpose = @x_matrix.transpose
    y_matrix = @y.map { |y| Array(y) }
    # @a_array = ((Matrix[*@x_transpose] * Matrix[*@x_matrix]).inverse * (Matrix[*@x_transpose] * Matrix[*y_matrix])).to_a
    res = StatModels.OLS.new(y_matrix, x_other_matrix).fit
    @a_array = res.params.to_a
    @stds = res.cov_params.diagonal.to_a.map { |std| Math.sqrt(std) }
    @intervals = res.conf_int(0.05).to_a
    # @a_array.flatten!
    @stats = @a_array.map.with_index { |a, i| a / @stds[i] }
    @ps = @stats.map { |a_stat| 2 * (1 - ScipyStats.t(@count.pred).cdf(a_stat.abs)) }
    @dispersia = res.mse_resid
    @determination = res.rsquared
    mean_y = @y.sum.to_f / @count
    @f = res.fittedvalues.to_a.sum { |y| (y - mean_y)**2 }.to_f / @dispersia
    @p_f = 1 - ScipyStats.f(1, @count - 2).cdf(@f)
    @e_array = res.resid.to_a
    @e_assym = assymetry_stat
    @e_excess = excess_stat
    @diagram = diagram(res.fittedvalues, res.resid)
  end

  def spirmen
    ranks_x = ranks(@dataset_x)
    ranks_y = ranks(@dataset_y)
    ranks_different = ranks_x.map.with_index { |rank, index| (rank - ranks_y[index])**2 }.sum.to_f
    nn2_1 = (@count * ((@count**2) - 1)).to_f # rubocop:disable Naming/VariableNumber
    eq_x = equal_ranks(ranks_x)
    eq_y = equal_ranks(ranks_y)
    a = eq_x.sum { |_, aj| (aj**3) - aj }.to_f / 12
    b = eq_y.sum { |_, bk| (bk**3) - bk }.to_f / 12
    ((nn2_1 / 6) - ranks_different - a - b) / Math.sqrt(((nn2_1 / 6) - (2 * a)) * ((nn2_1 / 6) - (2 * b)))
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

  def assymetry_stat
    mean = @e_array.sum.to_f / @count
    std_moved = Math.sqrt(@e_array.sum { |value| (value - mean)**2 }.to_f / @count)
    assymetry = (Math.sqrt(@count * @count.pred) / (@count - 2)) * (@e_array.sum { |value| (value - mean)**3 }.to_f / @count / (std_moved**3))
    assymetry_std = Math.sqrt((6 * @count * @count.pred).to_f / ((@count - 2) * @count.next * (@count + 3)))
    assymetry / assymetry_std
  end

  def excess_stat
    mean = @e_array.sum.to_f / @count
    std_moved = Math.sqrt(@e_array.sum { |value| (value - mean)**2 }.to_f / @count)
    excess_moved = (@e_array.sum { |value| (value - mean)**4 }.to_f / @count / (std_moved**4)) - 3
    excess = ((@count**2).pred.to_f / (@count - 2) / (@count - 3)) * (excess_moved + (6.0 / @count.next))
    excess_std = Math.sqrt((24 * @count * (@count.pred**2)).to_f / ((@count - 2) * (@count - 3) * (@count + 3) * (@count + 5)))
    excess / excess_std
  end

  def correlation_field
    PLT.scatter(@dependent_column, @independent_column)
    PLT.ylabel('y')
    PLT.xlabel('x')
    PLT.plot(@dependent_column, @dependent_column.map { |x| @a0 + (@a1 * x) }, color: 'green')
    fn = @dependent_column.zip(@fxn).sort_by(&:first)
    fv = @dependent_column.zip(@fxv).sort_by(&:first)
    yn = @dependent_column.zip(@yxn).sort_by(&:first)
    yv = @dependent_column.zip(@yxv).sort_by(&:first)
    PLT.plot(fn.map(&:first), fn.map(&:last), color: 'orange')
    PLT.plot(fv.map(&:first), fv.map(&:last), color: 'orange')
    PLT.plot(yn.map(&:first), yn.map(&:last), color: 'red')
    PLT.plot(yv.map(&:first), yv.map(&:last), color: 'red')
    plot_image
  end

  def diagram(value_y, value_e)
    PLT.scatter(value_y, value_e)
    PLT.ylabel('e')
    PLT.xlabel('y')
    plot_image
  end

  def pearson(values_x, values_y)
    mean_xy = values_x.zip(values_y).sum { |x, y| x * y }.to_f / @count
    std_moved_x = Math.sqrt(values_x.sum { |value| (value - @mean_x)**2 }.to_f / @count)
    std_moved_y = Math.sqrt(values_y.sum { |value| (value - @mean_y)**2 }.to_f / @count)
    r = (mean_xy - (@mean_x * @mean_y)) / (std_moved_x * std_moved_y)
    { r: r,
      std_moved_x: std_moved_x,
      std_moved_y: std_moved_y }
  end

  def emp_params
    params.permit(:data, :dependent_indicator, :independent_indicator, :part)
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
