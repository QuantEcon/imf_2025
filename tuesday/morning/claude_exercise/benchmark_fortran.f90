program benchmark_fortran
    implicit none
    integer, parameter :: n = 100001
    real(8), parameter :: x0 = 0.3d0
    real(8), dimension(n) :: x
    real(8) :: start_time, end_time, elapsed
    integer :: t

    ! Warm-up run
    call generate_time_series(x0, 1000, x)

    ! Timed run
    call cpu_time(start_time)
    call generate_time_series(x0, n, x)
    call cpu_time(end_time)

    elapsed = end_time - start_time

    print '(A,I0,A,F10.6,A)', 'Fortran: Generated ', n, ' observations in ', elapsed, ' seconds'
    print '(A,F0.0,A)', 'Fortran: ', real(n) / elapsed, ' iterations per second'

contains

    ! The quadratic map g(x) = 4x(1-x)
    function g(x) result(y)
        real(8), intent(in) :: x
        real(8) :: y
        y = 4.0d0 * x * (1.0d0 - x)
    end function g

    ! Generate n observations from the quadratic map
    subroutine generate_time_series(x0, n, x)
        real(8), intent(in) :: x0
        integer, intent(in) :: n
        real(8), dimension(n), intent(out) :: x
        integer :: t

        x(1) = x0
        do t = 1, n - 1
            x(t + 1) = g(x(t))
        end do
    end subroutine generate_time_series

end program benchmark_fortran
