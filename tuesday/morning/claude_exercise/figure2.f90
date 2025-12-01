program figure2
    implicit none
    integer, parameter :: n_steps = 150
    real(8) :: x0, x_current
    real(8), dimension(0:n_steps) :: trajectory
    integer :: t

    ! Initial condition
    x0 = 0.3d0
    trajectory(0) = x0
    x_current = x0

    ! Generate time series using quadratic map g(x) = 4x(1-x)
    do t = 1, n_steps
        x_current = 4.0d0 * x_current * (1.0d0 - x_current)
        trajectory(t) = x_current
    end do

    ! Write to file
    open(unit=10, file='figure2_data.txt', status='replace')
    write(10, '(A)') '# t x_t'
    do t = 0, n_steps
        write(10, '(I10, F20.10)') t, trajectory(t)
    end do
    close(10)

    print *, 'Figure 2 data written to figure2_data.txt'

end program figure2
