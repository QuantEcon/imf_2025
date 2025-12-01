program figure3
    implicit none
    integer, parameter :: n_steps = 100000
    real(8) :: x0, x_current
    real(8), dimension(0:n_steps) :: trajectory
    integer :: t
    real(8) :: start_time, end_time, elapsed_time
    integer :: count_rate, count_max
    integer :: start_count, end_count

    ! Initial condition
    x0 = 0.3d0
    trajectory(0) = x0
    x_current = x0

    ! Start timer
    call system_clock(start_count, count_rate, count_max)

    ! Generate long time series using quadratic map g(x) = 4x(1-x)
    do t = 1, n_steps
        x_current = 4.0d0 * x_current * (1.0d0 - x_current)
        trajectory(t) = x_current
    end do

    ! End timer
    call system_clock(end_count)
    elapsed_time = real(end_count - start_count) / real(count_rate)

    ! Write to file
    open(unit=10, file='figure3_data.txt', status='replace')
    write(10, '(A)') '# x_t'
    do t = 0, n_steps
        write(10, '(F20.10)') trajectory(t)
    end do
    close(10)

    print *, 'Figure 3 data written to figure3_data.txt'
    print *, 'Total observations:', n_steps + 1
    print '(A,F10.6,A)', 'Execution time: ', elapsed_time, ' seconds'

end program figure3
