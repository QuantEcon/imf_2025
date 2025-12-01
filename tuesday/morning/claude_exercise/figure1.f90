program figure1
    implicit none
    integer, parameter :: n_points = 1000
    real(8) :: x, dx
    real(8), dimension(n_points) :: x_vals, g_vals, diagonal
    integer :: i

    ! Generate x values from 0 to 1
    dx = 1.0d0 / dble(n_points - 1)
    do i = 1, n_points
        x_vals(i) = dble(i - 1) * dx
        ! Calculate g(x) = 4x(1-x)
        g_vals(i) = 4.0d0 * x_vals(i) * (1.0d0 - x_vals(i))
        ! 45 degree line
        diagonal(i) = x_vals(i)
    end do

    ! Write to file
    open(unit=10, file='figure1_data.txt', status='replace')
    write(10, '(A)') '# x g(x) diagonal'
    do i = 1, n_points
        write(10, '(3F20.10)') x_vals(i), g_vals(i), diagonal(i)
    end do
    close(10)

    print *, 'Figure 1 data written to figure1_data.txt'

end program figure1
